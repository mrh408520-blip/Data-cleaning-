import os
import pandas as pd
import numpy as np
import webview
import json

class DataCleanerAPI:
    def __init__(self):
        self.df = None
        self.default_input_file = None
        self.default_output_name = 'output.csv'
        self.default_output_dir = ''
        self.use_save_dialog = True
        self.use_file_dialog = True

    def _get_default_output_path(self):
        if self.default_output_dir:
            return os.path.join(self.default_output_dir, self.default_output_name)
        return self.default_output_name

    def _safe_preview(self, df):
        preview = df.head(50).copy().astype(object)
        preview = preview.where(pd.notnull(preview), None)

        records = []
        for row in preview.to_dict(orient='records'):
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (np.ndarray,)):
                    value = value.tolist()
                clean_row[key] = None if pd.isna(value) else value
            records.append(clean_row)
        return records

    # NEW: file dialog handler
    def open_file_dialog(self):
        try:
            if not self.use_file_dialog:
                return None

            file_types = ('CSV Files (*.csv)', 'Excel Files (*.xlsx;*.xls)')
            dialog_type = getattr(webview, 'FileDialog', None)
            if dialog_type is not None:
                dialog_type = webview.FileDialog.OPEN
            else:
                dialog_type = webview.OPEN_DIALOG

            result = webview.windows[0].create_file_dialog(dialog_type, allow_multiple=False, file_types=file_types)

            if result and len(result) > 0:
                return result[0]
            return None
        except Exception:
            return None

    def load_file(self, file_path):
        try:
            if not file_path or not isinstance(file_path, str):
                if self.default_input_file:
                    file_path = self.default_input_file
                else:
                    return {'error': 'Invalid file path'}

            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file_path)
            else:
                return {'error': 'Unsupported file format'}

            return {
                'columns': list(self.df.columns),
                'preview': self._safe_preview(self.df)
            }
        except Exception as e:
            return {'error': str(e)}

    def clean_data(self, config_json):
        try:
            if self.df is None:
                return {'error': 'No data loaded'}

            config = json.loads(config_json) if isinstance(config_json, str) else config_json
            if not isinstance(config, dict):
                return {'error': 'Invalid config'}

            if config.get('remove_nulls'):
                self.df = self.df.dropna()

            if config.get('drop_duplicates'):
                self.df = self.df.drop_duplicates()

            normalize_cols = config.get('normalize_text', [])
            if isinstance(normalize_cols, str):
                normalize_cols = [normalize_cols]

            for col in normalize_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).str.lower().str.strip()

            return {
                'columns': list(self.df.columns),
                'preview': self._safe_preview(self.df)
            }
        except Exception as e:
            return {'error': str(e)}

    def save_file_dialog(self, default_name='output.csv'):
        try:
            if not self.use_save_dialog:
                return self._get_default_output_path()

            file_types = ('CSV Files (*.csv)', 'Excel Files (*.xlsx;*.xls)')
            dialog_type = getattr(webview, 'FileDialog', None)
            if dialog_type is not None:
                dialog_type = webview.FileDialog.SAVE
            else:
                dialog_type = webview.SAVE_DIALOG

            result = webview.windows[0].create_file_dialog(dialog_type, allow_multiple=False, file_types=file_types, save_filename=default_name)

            if result and len(result) > 0:
                return result[0]
            return None
        except Exception:
            return None

    def export_file(self, file_path):
        try:
            if self.df is None:
                return {'error': 'No data to export'}

            if not file_path or not isinstance(file_path, str):
                return {'error': 'Invalid export path'}

            if file_path.endswith('.csv'):
                self.df.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                self.df.to_excel(file_path, index=False)
            else:
                return {'error': 'Unsupported export format'}

            return {'status': 'success'}
        except Exception as e:
            return {'error': str(e)}


api = DataCleanerAPI()

if __name__ == '__main__':
    window = webview.create_window(
        'Data Cleaner',
        'https://data-cleaning-45.netlify.app/',
        js_api=api,
        width=1000,
        height=700
    )

    webview.start()