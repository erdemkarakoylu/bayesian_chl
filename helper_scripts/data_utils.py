import numpy as np
from pandas import DataFrame
import re

# Define the mapping from band number to wavelength (in nm) for each sensor
band_wavelength_mapping = {
    'MODIS': {1: 412, 2: 443, 3: 486, 4: 547, 5: 555, 6: 667, 7: 678, 8: 748, 9: 869},
    'MERIS': {1: 412.5, 2: 442.5, 3: 490, 4: 510, 5: 560, 6: 620, 7: 665, 8: 681.25, 
              9: 708.75, 10: 753.75, 11: 760.625, 12: 778.75, 13: 865, 14: 885, 15: 900},
    'SeaWiFS': {1: 412, 2: 443, 3: 490, 4: 510, 5: 555, 6: 670, 7: 765, 8: 865},
    'VIIRS-SNPP': {1: 410, 2: 443, 3: 486, 4: 551, 5: 671, 6: 745, 7: 862},
    'VIIRS-JPSS': {1: 410, 2: 443, 3: 486, 4: 551, 5: 671, 6: 745, 7: 862},
    'OLCI-S3A': {1: 400, 2: 412.5, 3: 442.5, 4: 490, 5: 510, 6: 560, 7: 620, 8: 665, 
                 9: 673.75, 10: 681.25, 11: 708.25, 12: 753.75, 13: 761.25, 14: 764.375, 
                 15: 767.5, 16: 778.75, 17: 865, 18: 885, 19: 900, 20: 940, 21: 1020},
    'OLCI-S3B': {1: 400, 2: 412.5, 3: 442.5, 4: 490, 5: 510, 6: 560, 7: 620, 8: 665, 
                 9: 673.75, 10: 681.25, 11: 708.25, 12: 753.75, 13: 761.25, 14: 764.375, 
                 15: 767.5, 16: 778.75, 17: 865, 18: 885, 19: 900, 20: 940, 21: 1020}
}

# Define the color category mapping
band_color_mapping = {
    'Indigo': [412, 410],
    'Blue': [443, 442.5, 443],
    'Cyan': [490, 486, 490, 510],
    'Green': [510, 547, 551, 560],
    'Yellow': [555],
    'Orange': [620, 617],
    'Red': [665, 667, 670, 673.75, 681],
    'Near-IR': [753.75, 765, 869, 862]
}

def extract_sensor_and_band(col_name: str):
    """
    Extracts the sensor abbreviation and band number from a column name.
    
    Args:
    - col_name (str): The column name (e.g., 'rrs_MOD1', 'rrs_OLCA1')
    
    Returns:
    - sensor_name (str): The sensor abbreviation (e.g., 'MODIS', 'OLCI-S3A')
    - band_number (int): The band number (e.g., 1, 2, etc.)
    """
    # Use regex to separate the sensor abbreviation and the band number
    match = re.match(r"([A-Za-z]+)(\d+)$", col_name.split('_')[1])
    if match:
        sensor_name = match.group(1)  # Sensor abbreviation (letters)
        band_number = int(match.group(2))  # Band number (digits)
        return sensor_name, band_number
    else:
        return None, None  # Return None if no match

def get_wavelength(sensor_name: str, band_number: int):
    """
    Retrieves the corresponding wavelength for a given band number and sensor.
    
    Args:
    - sensor_name (str): The sensor abbreviation (e.g., 'MODIS', 'OLCI-S3A')
    - band_number (int): The band number (e.g., 1, 2, etc.)
    
    Returns:
    - wavelength (float): The wavelength for the band in nm
    """
    if band_number in band_wavelength_mapping[sensor_name]:
        return band_wavelength_mapping[sensor_name][band_number]
    else:
        return None

def get_color_category(wavelength: float):
    """
    Determines the color category based on the wavelength.
    
    Args:
    - wavelength (float): The wavelength in nm
    
    Returns:
    - color_category (str): The color category (e.g., 'Blue', 'Red', 'Near-IR')
    """
    for color, wavelengths in band_color_mapping.items():
        if wavelength in wavelengths:
            return color
    return None

def reorganize_rrs_columns(df: DataFrame):
    """
    Reorganizes the remote sensing reflectance (Rrs) columns into color categories 
    based on sensor abbreviation and band number.
    
    Args:
    - df (pd.DataFrame): The input DataFrame with Rrs data (sensor bands as columns)
    
    Returns:
    - reorganized_df (pd.DataFrame): The transformed DataFrame with columns by color category and sensor
    """
    reorganized_df = DataFrame()
    
    for col in df.columns:
        # Only process columns that match the expected format (i.e., 'rrs_[sensor][band]')
        if not col.startswith('rrs_'):
            print(f"Skipping column {col}: does not match expected 'rrs_[sensor][band]' format.")
            continue
        
        # Extract the sensor and band number
        sensor_name, band_number = extract_sensor_and_band(col)
        
        # Skip if extraction failed (invalid column format)
        if sensor_name is None or band_number is None:
            print(f"Skipping column {col}: Invalid sensor and band format.")
            continue
        
        # Map sensor abbreviations to full names
        sensor_name = {
            'MOD': 'MODIS',
            'MER': 'MERIS',
            'SWS': 'SeaWiFS',
            'VIRPP': 'VIIRS-SNPP',
            'VIRJP': 'VIIRS-JPSS',
            'OLCA': 'OLCI-S3A',
            'OLCB': 'OLCI-S3B'
        }.get(sensor_name, sensor_name)  # Default to the sensor name if it's valid
        
        # Get the wavelength for the band
        wavelength = get_wavelength(sensor_name, band_number)
        
        # Skip if wavelength is not found
        if wavelength is None:
            print(f"Skipping column {col}: Band {band_number} not found in wavelength mapping.")
            continue
        
        # Get the color category
        color_category = get_color_category(wavelength)
        
        # Skip if color category is not found
        if color_category is None:
            print(f"Skipping column {col}: No color category found for wavelength {wavelength}.")
            continue
        
        # Add the color category and sensor name to the reorganized DataFrame
        new_col_name = f'{color_category}_{sensor_name}'
        reorganized_df[new_col_name] = df[col]
        reorganized_df['Sensor'] = sensor_name

    return reorganized_df

# Example usage:
# Assuming `df_rrs` is your original DataFrame with Rrs data
# df_rrs_transformed = reorganize_rrs_columns(df_rrs)



def create_chl_and_flag(
        df:DataFrame, hplc_label:str='chl_a', 
        fluo_label:str='chl') -> DataFrame:
    """
    Creates 'chl' and 'hplc_flag' columns based on the rules provided.

    Args:
        df (pd.DataFrame): Input DataFrame with fluo_label and hplc_label columns.

    Returns:
        pd.DataFrame: DataFrame with new 'chl' and 'hplc_flag' columns.
    """
    new_chl = np.select(
        [
            df[hplc_label].notna(),
            df[fluo_label].notna() & df[hplc_label].isna(),
            df[fluo_label].isna() & df[hplc_label].isna()
        ],
        [
            df[hplc_label],
            df[fluo_label],
            np.nan
        ],
        default=np.nan  # Should not be reached based on the conditions
    )

    hplc_flag = np.select(
        [
            df[hplc_label].notna(),
            df[fluo_label].notna() & df[hplc_label].isna(),
            df[fluo_label].isna() & df[hplc_label].isna()
        ],
        ['hplc', 'fluo', 'None'],
        default='None' # Should not be reached based on the conditions
    )

    new_df = df.copy()
    new_df['chl'] = new_chl
    new_df['hplc_flag'] = hplc_flag
    return new_df[['chl', 'hplc_flag']]


def summarize_idata(idata, name: str):
    print(f"\n{name}")
    print(idata)
    for grp in ["posterior", "log_likelihood", "posterior_predictive", "observed_data"]:
        has = 'Yes' if hasattr(idata, grp) else 'No'
        print(f"  has {grp}: {has}")