import pandas as pd

def extract_mean(value):
    """Extract mean value from string like '0.6774±0.1230'"""
    if isinstance(value, str) and '±' in value:
        return float(value.split('±')[0])
    return value


def read_results_table(path_to_table, column_to_read):
    """
    Read a specific column from a xlsx table and create dictionaries for two models.
    """
    df = pd.read_excel(path_to_table)
    
    omaseg_col = 'OMASeg ' + column_to_read
    totalseg_col = 'TotalSeg ' + column_to_read
    
    if omaseg_col not in df.columns or totalseg_col not in df.columns:
        raise ValueError(f"Columns '{omaseg_col}' and/or '{totalseg_col}' not found in table. "
                        f"Available columns: {', '.join(df.columns)}")
    
    if 'mean' in column_to_read.lower():
        omaseg_results = {organ: extract_mean(value) 
                        for organ, value in zip(df['Organ'], df[omaseg_col])}
        totalseg_results = {organ: extract_mean(value) 
                            for organ, value in zip(df['Organ'], df[totalseg_col])}
    else:
        omaseg_results = dict(zip(df['Organ'], df[omaseg_col]))
        totalseg_results = dict(zip(df['Organ'], df[totalseg_col]))
    
    if not omaseg_results or not totalseg_results:
        raise ValueError("No data found in the specified columns")
        
    return omaseg_results, totalseg_results

# check: up-to-date with my system definition?
anatomical_system_colors = {
    "Skeletal System": "#4DAF7C",
    "Muscular System": "#AA4A44",
    "Cardiovascular System": "#D03B3B",
    "Respiratory System": "#7EB5A6",
    "Digestive System": "#B87D4B",
    "Urinary System": "#EFC75E",
    "Reproductive System": "#E091C3",
    "Endocrine System": "#5C9BD1",
    "Nervous System": "#8E63B5",
    "Sensory Organs": "#98D8AA",
    "Lymphatic and Immune System": "#5E97A3",
    "Body Cavities": "#9A8A76",
    "Other Structures": "#808080",
    "Glandular System": "#E39A2E"
}