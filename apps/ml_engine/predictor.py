import math
from datetime import datetime, timedelta

import joblib
import pandas as pd
from django.conf import settings
from django.utils import timezone

from apps.core.models import Asset, Prediction
from .dataset_builder import DatasetBuilder
from .trainer import CLASS_LABELS, model_path_for


class PredictionUnavailable(RuntimeError):
    """No es posible emitir una predicción respaldada por un modelo entrenado."""


def next_target_date(asset_type: str, last_bar: datetime) -> datetime:
    """
    Fecha objetivo = siguiente barra diaria a partir de la última barra usada.

    - Cripto: +24 h (el mercado opera todos los días).
    - Acciones: siguiente día hábil (lunes a viernes). No contempla feriados
      bursátiles; para el MVP no se añade dependencia de calendario de mercado.
    """
    if asset_type == Asset.AssetType.CRYPTO:
        return last_bar + timedelta(days=1)
    target = last_bar + timedelta(days=1)
    while target.weekday() >= 5:  # 5 = sábado, 6 = domingo
        target += timedelta(days=1)
    return target


class MLPredictor:
    def __init__(self, version: str = "v1.0.0"):
        self.version = version
        self.artifact = None

    @property
    def model_file(self):
        return model_path_for(self.version)

    def load_model(self) -> dict:
        """Carga el artefacto entrenado; falla explícitamente si no existe o es incompatible."""
        if self.artifact is not None:
            return self.artifact
        if not self.model_file.exists():
            raise PredictionUnavailable(
                f"No existe el modelo entrenado '{self.version}' ({self.model_file.name}). "
                f"Ejecute train_ml_model --model-version {self.version}."
            )
        artifact = joblib.load(self.model_file)
        if not isinstance(artifact, dict) or 'model' not in artifact or 'features' not in artifact:
            raise PredictionUnavailable(f"El artefacto '{self.model_file.name}' no es compatible.")
        self.artifact = artifact
        return artifact

    def predict_asset(self, asset: Asset) -> Prediction:
        """
        Genera la predicción probabilística (SUBE, ESTABLE, BAJA) para un activo usando
        las features de la última barra disponible en T. Nunca persiste una clase que no
        provenga de un modelo entrenado.
        """
        artifact = self.load_model()
        feature_cols = list(artifact['features'])

        df_asset = DatasetBuilder(asset).build_dataset(forward_days=1)
        if df_asset.empty:
            raise PredictionUnavailable(
                f"{asset.ticker}: sin datos de mercado. Ejecute fetch_market_data."
            )

        latest_row = df_asset.iloc[-1]
        last_bar = pd.Timestamp(latest_row['timestamp']).to_pydatetime()
        if timezone.is_naive(last_bar):
            last_bar = timezone.make_aware(last_bar)

        max_age = timedelta(days=float(getattr(settings, 'ML_MAX_DATA_AGE_DAYS', 7)))
        if timezone.now() - last_bar > max_age:
            raise PredictionUnavailable(
                f"{asset.ticker}: la última barra ({last_bar:%Y-%m-%d}) supera la antigüedad "
                f"máxima de {max_age.days} días. Actualice con fetch_market_data."
            )

        missing = [c for c in feature_cols if c not in df_asset.columns]
        if missing:
            raise PredictionUnavailable(
                f"{asset.ticker}: el dataset no contiene las features del modelo: {', '.join(missing)}"
            )

        X_sample = pd.DataFrame([latest_row[feature_cols]]).astype(float).fillna(0.0)
        model = artifact['model']
        probs = model.predict_proba(X_sample)[0]
        prob_map = {str(cls): float(p) for cls, p in zip(model.classes_, probs)}
        p_sube, p_estable, p_baja = (prob_map.get(label, 0.0) for label in CLASS_LABELS)

        total = p_sube + p_estable + p_baja
        if not math.isfinite(total) or total <= 0:
            raise PredictionUnavailable(f"{asset.ticker}: el modelo devolvió probabilidades no válidas.")
        p_sube, p_estable, p_baja = p_sube / total, p_estable / total, p_baja / total

        normalized = dict(zip(CLASS_LABELS, (p_sube, p_estable, p_baja)))
        predicted_class = max(normalized, key=normalized.get)

        return Prediction.objects.create(
            asset=asset,
            target_date=next_target_date(asset.asset_type, last_bar),
            predicted_class=predicted_class,
            prob_sube=p_sube,
            prob_estable=p_estable,
            prob_baja=p_baja,
            model_version=self.version,
        )
