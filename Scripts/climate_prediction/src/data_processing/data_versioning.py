import json
from datetime import datetime

class DataVersioning:
    def __init__(self, output_dir='outputs/metadata'):
        self.output_dir = output_dir
    
    def save_version_info(self, df, preprocessing_steps):
        """Save versioning information about the processed dataset."""
        version_info = {
            'timestamp': datetime.now().isoformat(),
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'preprocessing_steps': preprocessing_steps,
            'data_stats': {
                'start_date': df.index.min().isoformat(),
                'end_date': df.index.max().isoformat()
            }
        }
        
        version_file = f"{self.output_dir}/data_version_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=4)
        
        return version_info