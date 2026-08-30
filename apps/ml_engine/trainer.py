import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from apps.core.models import Asset, ModelMetrics
from .dataset_builder import DatasetBuilder

# Directorio predeterminado de artefactos para modelos entrenados
ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'


def get_artifacts_dir() -> Path:
    """Directorio de artefactos (configurable con settings.ML_ARTIFACTS_DIR)."""
    path = Path(getattr(settings, 'ML_ARTIFACTS_DIR', None) or ARTIFACTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path_for(version: str) -> Path:
    return get_artifacts_dir() / f'model_{version}.joblib'

CLASS_LABELS = ['SUBE', 'ESTABLE', 'BAJA']


class MLTrainer:
    FEATURE_COLS = [
        'close_price', 'volume', 'rsi_14', 'macd', 'macd_signal',
        'sma_20', 'sma_50', 'volatility_20',
        'news_sentiment_score', 'news_impact_score', 'news_count'
    ]
    # Mínimo de fechas distintas etiquetadas para poder validar temporalmente.
    MIN_UNIQUE_DATES = 10
    MAX_SPLITS = 3
    RANDOM_STATE = 42

    def __init__(self, version: str = "v1.0.0"):
        self.version = version

    @classmethod
    def _new_model(cls) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=100, max_depth=6, random_state=cls.RANDOM_STATE)

    @staticmethod
    def temporal_folds(timestamps: pd.Series, n_splits: int):
        """
        Genera folds (train_idx, val_idx) sobre fechas de calendario completas (UTC).

        Las filas de distintos activos que comparten fecha quedan siempre del mismo
        lado del corte, de modo que ningún día aparece en entrenamiento y validación
        a la vez. Se mantiene TimeSeriesSplit sobre las fechas únicas ordenadas.
        """
        dates = pd.to_datetime(timestamps, utc=True).dt.date.reset_index(drop=True)
        unique_dates = np.array(sorted(dates.unique()))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_d, val_d in tscv.split(unique_dates):
            train_mask = dates.isin(set(unique_dates[train_d])).to_numpy()
            val_mask = dates.isin(set(unique_dates[val_d])).to_numpy()
            yield np.flatnonzero(train_mask), np.flatnonzero(val_mask)

    def _collect_dataset(self) -> pd.DataFrame:
        all_dfs = []
        for asset in Asset.objects.filter(is_active=True):
            labeled = DatasetBuilder.labeled_rows(
                DatasetBuilder(asset).build_dataset(forward_days=1)
            )
            if not labeled.empty:
                all_dfs.append(labeled)
        if not all_dfs:
            return pd.DataFrame()
        df_full = pd.concat(all_dfs, ignore_index=True)
        return df_full.sort_values('timestamp', kind='mergesort').reset_index(drop=True)

    def train_and_evaluate(self) -> dict:
        """
        Consolida datasets de todos los activos activos, valida con folds temporales
        por fecha completa, calcula métricas fuera de muestra y luego entrena el
        artefacto final con todas las filas etiquetadas.
        """
        df_full = self._collect_dataset()
        if df_full.empty:
            return {'status': 'error', 'message': 'Sin datos etiquetados para entrenamiento. '
                    'Ejecute fetch_market_data primero.'}

        n_dates = pd.to_datetime(df_full['timestamp'], utc=True).dt.date.nunique()
        if n_dates < self.MIN_UNIQUE_DATES:
            return {
                'status': 'error',
                'message': (f'Insuficientes fechas etiquetadas ({n_dates} < {self.MIN_UNIQUE_DATES}) '
                            'para validación temporal. Descargue más histórico de mercado.'),
            }

        X = df_full[self.FEATURE_COLS].astype(float).fillna(0.0)
        y = df_full['target'].astype(str)

        n_splits = min(self.MAX_SPLITS, max(2, n_dates // 5))
        oof_true, oof_pred = [], []
        for train_idx, val_idx in self.temporal_folds(df_full['timestamp'], n_splits):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            clf = self._new_model()
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_true.extend(y.iloc[val_idx].tolist())
            oof_pred.extend(clf.predict(X.iloc[val_idx]).tolist())

        if not oof_true:
            return {'status': 'error', 'message': 'No se pudo construir ningún fold de validación temporal.'}

        # Métricas exclusivamente fuera de muestra (predicciones de validación).
        accuracy = float(accuracy_score(oof_true, oof_pred))
        f1 = float(f1_score(oof_true, oof_pred, average='weighted', zero_division=0))
        cm = confusion_matrix(oof_true, oof_pred, labels=CLASS_LABELS).tolist()

        # Artefacto final entrenado con todas las filas etiquetadas.
        final_model = self._new_model()
        final_model.fit(X, y)

        model_path = model_path_for(self.version)
        joblib.dump({
            'model': final_model,
            'features': self.FEATURE_COLS,
            'classes': list(final_model.classes_),
        }, model_path)

        ModelMetrics.objects.update_or_create(
            model_version=self.version,
            defaults={
                'accuracy': accuracy,
                'f1_score': f1,
                'confusion_matrix': cm,
                'features_used': self.FEATURE_COLS,
            }
        )

        return {
            'status': 'success',
            'model_version': self.version,
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'samples_trained': len(df_full),
            'validation_samples': len(oof_true),
            'unique_dates': int(n_dates),
            'n_splits': n_splits,
            'model_file': str(model_path),
        }
