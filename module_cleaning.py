import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


__all__ = ['geocode_addresses', 'LLM_treatment']

def geocode_addresses(df_pre_cleaned, gmaps_client, location_column="Location", rate_limit=0.02):
    def extract_components(messy_address):
        if pd.isna(messy_address) or not str(messy_address).strip():
            return None, None, None, "Blank Input"  
        
        try:
            clean_input = str(messy_address).encode('latin1', errors='ignore').decode('utf-8', errors='ignore').strip()
            
            if not clean_input:
                clean_input = str(messy_address)
            
            geocode_result = gmaps_client.geocode(address=f"{clean_input}, Philippines", language="en")
            
            if not geocode_result:
                return None, None, None, "No Match Found" 
            
            result_data = geocode_result[0]
            components = result_data.get("address_components", [])
            formatted_address = result_data.get("formatted_address", "Not Found")
            barangay = city_municipality = province = None 
            google_region = None  #
            
            for component in components:
                types = component.get("types", [])
                long_name = component.get("long_name")
                if "administrative_area_level_1" in types:
                    google_region = long_name
                elif "administrative_area_level_2" in types:
                    province = long_name
                elif "locality" in types:
                    city_municipality = long_name
                elif any(t in types for t in ["neighborhood", "sublocality", "sublocality_level_1", "sublocality_level_2", "colloquial_area"]):
                    barangay = long_name
            
            if google_region == "National Capital Region" or "Metro Manila" in formatted_address:
                if province is None or province == "Not Found":  
                    province = "Metro Manila"
            
            if province == city_municipality and google_region is not None and google_region != "Not Found":
                province = google_region
            
            if barangay in (city_municipality, province):
                barangay = None 
            
            return barangay, city_municipality, province, formatted_address
        except Exception:
            return None, None, None, "Error"  
    
    batch_records = []
    for idx, row in tqdm(df_pre_cleaned.iterrows(), total=len(df_pre_cleaned), desc="Geocoding Progress"):
        brgy, city, prov, full_addr = extract_components(row[location_column])
        row["Town/Barangay"] = brgy 
        row["Municipality/City"] = city 
        row["Province/NCR"] = prov  
        row["Google_Cleaned_Address"] = full_addr
        batch_records.append(row)
        time.sleep(rate_limit)
    
    return pd.DataFrame(batch_records)

def safe_parse_location(client, row):
    """Parse location using the provided OpenAI client."""
    prompt = f"""
ADDRESS PARSER - Extract Barangay, Municipality/City, and Province
LOCATION ADDRESS: {row['Location']}
GOOGLE ADDRESS: {row['Google_Cleaned_Address']}

TASK: Parse the address above and extract:
1. Barangay (neighborhood/village)
2. Municipality or City
3. Province

OUTPUT FORMAT: Barangay, Municipality/City, Province
- Separate with commas, no spaces
- From the information given, and clues parse the address. Look up the available data or information or 
knowledge that is known about to which barangay, municipality or city, province the Location Address 
and its Google Cleaned Address belongs
- Keep the ñ of the names. 
- Keep the consistent standard formal naming of places. 
- Any given in native Filipino name e.g. Kalakhang Maynila, Encode "Metro Manila";
 'Lalawigan ng Laguna' encode Laguna; 'Lalawigan ng Cebu' encode Cebu.; 'Makati' encode as Makati City
- If no any information, write "Not Found"
- DO NOT add any extra text or explanation
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
            top_p=0.1,
            extra_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "Location Parser"}
        )
        
        result = response.choices[0].message.content.strip()
        
        if result != "Not Found" and ',' in result:
            parts = [p.strip() for p in result.split(',')]
            if len(parts) >= 3:
                return {
                    'Town/Barangay': parts[0] if parts[0] != 'Not Found' else None,  # Renamed key
                    'Municipality/City': parts[1] if parts[1] != 'Not Found' else None,  # Renamed key
                    'Province/NCR': parts[2] if parts[2] != 'Not Found' else None  # Renamed key
                }
        
        return {'Town/Barangay': None, 'Municipality/City': None, 'Province/NCR': None}  # Renamed keys
    
    except Exception:
        return {'Town/Barangay': None, 'Municipality/City': None, 'Province/NCR': None}  # Renamed keys


def LLM_treatment(data_frame, client):
    df = data_frame.copy()
    
    for col in ['Town/Barangay', 'Municipality/City', 'Province/NCR']:
        if col not in df.columns:
            df[col] = None
    
    needs_fix = (df['Town/Barangay'].isna()) | (df['Municipality/City'].isna()) | (df['Province/NCR'].isna())
    has_data = df['Location'].notna() | df['Google_Cleaned_Address'].notna()
    rows_to_fix = df.index[needs_fix & has_data]
    
    for idx in tqdm(rows_to_fix, desc="LLM Processing", unit="row"):
        parsed = safe_parse_location(client, df.loc[idx])
        
        for field in ['Town/Barangay', 'Municipality/City', 'Province/NCR']:
            if pd.isna(df.loc[idx, field]):
                df.loc[idx, field] = parsed[field]
        
        time.sleep(0.3)
    
    return df.drop('Google_Cleaned_Address', axis=1)
