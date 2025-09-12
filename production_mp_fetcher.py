import json
import re
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from typing import List, Dict, Optional, Union
import os
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionMPDataFetcher:
    """Production-ready Materials Project data fetcher and augmentation tool"""
    
    def __init__(self, mp_api_key: str):
        """Initialize with Materials Project API key"""
        self.mp_api_key = mp_api_key
        self.mpr = MPRester(mp_api_key)
        
        # CORRECTED feature mapping based on actual MP API fields
        self.feature_mapping = {
            # Basic structural properties
            "density": "density",
            "volume": "volume", 
            "nsites": "nsites",
            "space_group": "symmetry",
            "crystal_system": "symmetry",
            
            # Thermodynamic properties
            "formation_energy_per_atom": "formation_energy_per_atom",
            "energy_above_hull": "energy_above_hull",
            "uncorrected_energy_per_atom": "uncorrected_energy_per_atom",
            
            # Electronic properties
            "band_gap": "band_gap",
            "is_metal": "is_metal",
            "is_magnetic": "is_magnetic",
            "total_magnetization": "total_magnetization",
            
            # Chemical composition
            "num_elements": "nelements",
            "composition": "composition",
            "elements": "elements",
            "chemsys": "chemsys",
            "formula_pretty": "formula_pretty",
            
            # Mechanical properties (CORRECTED - direct mapping)
            "bulk_modulus": "bulk_modulus",
            "shear_modulus": "shear_modulus",
            "universal_anisotropy": "universal_anisotropy",
            
            # Dielectric properties (CORRECTED - direct mapping)
            "dielectric_total": "e_total",
            "dielectric_ionic": "e_ionic", 
            "dielectric_electronic": "e_electronic",
            
            # Advanced properties (limited availability)
            "dos": "dos",
            "band_structure": "bandstructure"
        }
        
        # High-coverage features (recommended for most use cases)
        self.reliable_features = [
            "density", "volume", "nsites", "formation_energy_per_atom",
            "energy_above_hull", "band_gap", "is_metal", "num_elements", 
            "formula_pretty", "space_group", "crystal_system",
            "bulk_modulus", "shear_modulus"  # Added mechanical properties
        ]
    
    def read_existing_data(self, file_path: str) -> pd.DataFrame:
        """Read existing materials data from CSV file"""
        
        logger.info(f"Reading existing data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Try different file formats
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                # Default to CSV
                df = pd.read_csv(file_path)
            
            logger.info(f"✅ Successfully loaded {len(df)} materials from {file_path}")
            logger.info(f"   Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error reading data file: {e}")
            raise
    
    def identify_material_id_column(self, df: pd.DataFrame) -> str:
        """Automatically identify the Materials Project ID column"""
        
        # Common column names for MP IDs
        possible_columns = [
            'material_id', 'mp_id', 'mpid', 'mp-id', 'id', 
            'Materials_Project_ID', 'MP_ID', 'material-id'
        ]
        
        for col in possible_columns:
            if col in df.columns:
                # Check if values look like MP IDs
                sample_values = df[col].dropna().head(10)
                if any(str(val).startswith('mp-') for val in sample_values):
                    logger.info(f"✅ Found MP ID column: {col}")
                    return col
        
        # Look for columns with 'mp-' pattern
        for col in df.columns:
            sample_values = df[col].dropna().head(10)
            if any(str(val).startswith('mp-') for val in sample_values):
                logger.info(f"✅ Detected MP ID column: {col}")
                return col
        
        # Manual selection
        print("\n🔍 Available columns:")
        for i, col in enumerate(df.columns, 1):
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            print(f"  {i:2d}. {col:20s} (sample: {sample_val})")
        
        while True:
            try:
                choice = input(f"\nSelect MP ID column number (1-{len(df.columns)}): ")
                col_idx = int(choice) - 1
                if 0 <= col_idx < len(df.columns):
                    selected_col = df.columns[col_idx]
                    logger.info(f"✅ User selected MP ID column: {selected_col}")
                    return selected_col
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def read_feature_recommendations(self, rec_file: str = None) -> List[str]:
        """Read feature recommendations from file or use defaults"""
        
        if rec_file and os.path.exists(rec_file):
            logger.info(f"Reading feature recommendations from: {rec_file}")
            
            try:
                with open(rec_file, 'r') as f:
                    content = f.read()
                
                # Parse features from file
                if "Parsed Features:" in content:
                    lines = content.split("Parsed Features:")[1].strip().split('\n')
                    features = [line.strip()[2:].strip() for line in lines 
                              if line.strip().startswith('- ') and line.strip()[2:].strip()]
                else:
                    # Extract from numbered format
                    pattern = r'(\d+)\.\s*([a-zA-Z_]+)'
                    matches = re.findall(pattern, content)
                    features = [match[1].strip() for match in matches]
                
                if features:
                    # Validate that recommended features exist in our mapping
                    valid_features = [f for f in features if f in self.feature_mapping]
                    invalid_features = [f for f in features if f not in self.feature_mapping]
                    
                    if invalid_features:
                        logger.warning(f"⚠️ Invalid features in recommendations: {invalid_features}")
                    
                    if valid_features:
                        logger.info(f"✅ Found {len(valid_features)} valid recommended features: {valid_features}")
                        return valid_features
                    
            except Exception as e:
                logger.warning(f"⚠️ Error reading recommendation file: {e}")
        
        # Use reliable defaults
        logger.info(f"Using default reliable features: {self.reliable_features}")
        return self.reliable_features.copy()
    
    def validate_material_ids(self, material_ids: List[str]) -> List[str]:
        """Validate and clean Materials Project IDs"""
        
        logger.info(f"Validating {len(material_ids)} material IDs...")
        
        valid_ids = []
        invalid_ids = []
        
        for mp_id in material_ids:
            # Clean and standardize
            clean_id = str(mp_id).strip()
            
            # Add 'mp-' prefix if missing
            if not clean_id.startswith('mp-') and clean_id.replace('mp', '').replace('-', '').isdigit():
                clean_id = f"mp-{clean_id.replace('mp', '').replace('-', '')}"
            
            # Validate format
            if clean_id.startswith('mp-') and clean_id[3:].isdigit():
                valid_ids.append(clean_id)
            else:
                invalid_ids.append(mp_id)
        
        if invalid_ids:
            logger.warning(f"⚠️ Found {len(invalid_ids)} invalid MP IDs: {invalid_ids[:5]}...")
        
        logger.info(f"✅ {len(valid_ids)} valid material IDs ready for processing")
        return valid_ids
    
    def fetch_mp_data_batch(self, 
                           material_ids: List[str], 
                           features: List[str],
                           batch_size: int = 100) -> pd.DataFrame:
        """Fetch MP data in batches for efficiency"""
        
        logger.info(f"Fetching MP data for {len(material_ids)} materials in batches of {batch_size}")
        
        # Map features to MP fields
        mp_fields = ['material_id']  # Always include material_id
        for feature in features:
            if feature in self.feature_mapping:
                mp_field = self.feature_mapping[feature]
                if mp_field not in mp_fields:
                    mp_fields.append(mp_field)
        
        logger.info(f"MP fields to fetch: {mp_fields}")
        
        all_data = []
        failed_batches = []
        
        # Process in batches
        for i in range(0, len(material_ids), batch_size):
            batch_ids = material_ids[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(material_ids) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_ids)} materials)")
            
            try:
                # Fetch data for this batch
                docs = self.mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=mp_fields
                )
                
                # Convert to list of dicts
                for doc in docs:
                    row_data = {'material_id': str(doc.material_id)}
                    
                    for field in mp_fields[1:]:  # Skip material_id
                        if hasattr(doc, field):
                            value = getattr(doc, field)
                            
                            # Handle special nested fields
                            if field == 'symmetry' and value:
                                row_data['space_group'] = getattr(value, 'symbol', None)
                                row_data['crystal_system'] = getattr(value, 'crystal_system', None)
                            elif field == 'nelements':
                                # Map nelements to num_elements for consistency
                                row_data['num_elements'] = value
                            elif field == 'bulk_modulus':
                                # Handle bulk modulus dictionary format
                                if isinstance(value, dict):
                                    row_data['bulk_modulus_voigt'] = value.get('voigt', None)
                                    row_data['bulk_modulus_reuss'] = value.get('reuss', None)
                                    row_data['bulk_modulus_vrh'] = value.get('vrh', None)
                                    row_data['bulk_modulus'] = value.get('vrh', None)  # Use VRH as default
                                else:
                                    row_data['bulk_modulus'] = value
                            elif field == 'shear_modulus':
                                # Handle shear modulus dictionary format
                                if isinstance(value, dict):
                                    row_data['shear_modulus_voigt'] = value.get('voigt', None)
                                    row_data['shear_modulus_reuss'] = value.get('reuss', None)
                                    row_data['shear_modulus_vrh'] = value.get('vrh', None)
                                    row_data['shear_modulus'] = value.get('vrh', None)  # Use VRH as default
                                else:
                                    row_data['shear_modulus'] = value
                            elif field == 'universal_anisotropy':
                                # Direct mapping for anisotropy
                                row_data[field] = value
                            elif field in ['e_total', 'e_ionic', 'e_electronic']:
                                # Map dielectric fields with readable names
                                field_mapping = {
                                    'e_total': 'dielectric_total',
                                    'e_ionic': 'dielectric_ionic', 
                                    'e_electronic': 'dielectric_electronic'
                                }
                                row_data[field_mapping[field]] = value
                            else:
                                row_data[field] = value
                        else:
                            row_data[field] = None
                    
                    all_data.append(row_data)
                
                logger.info(f"✅ Batch {batch_num} completed: {len(docs)} materials retrieved")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                failed_batches.append((batch_num, batch_ids))
                continue
        
        # Report results
        if failed_batches:
            logger.warning(f"⚠️ {len(failed_batches)} batches failed")
        
        if not all_data:
            logger.error("❌ No data retrieved from Materials Project")
            return pd.DataFrame()
        
        # Convert to DataFrame
        mp_df = pd.DataFrame(all_data)
        logger.info(f"✅ Successfully retrieved data for {len(mp_df)} materials")
        logger.info(f"   Columns: {list(mp_df.columns)}")
        
        return mp_df
    
    def merge_with_existing_data(self, 
                                existing_df: pd.DataFrame,
                                mp_df: pd.DataFrame, 
                                mp_id_column: str) -> pd.DataFrame:
        """Merge MP data with existing dataset"""
        
        logger.info("Merging MP data with existing dataset...")
        
        # Ensure MP ID columns have same name
        if mp_id_column != 'material_id':
            existing_df = existing_df.rename(columns={mp_id_column: 'material_id'})
        
        # Merge datasets
        merged_df = existing_df.merge(
            mp_df, 
            on='material_id', 
            how='left',
            suffixes=('', '_mp')
        )
        
        # Report merge statistics
        total_original = len(existing_df)
        total_merged = len(merged_df)
        materials_with_mp_data = merged_df['material_id'].isin(mp_df['material_id']).sum()
        
        logger.info(f"📊 Merge Statistics:")
        logger.info(f"   Original materials: {total_original}")
        logger.info(f"   After merge: {total_merged}")
        logger.info(f"   With MP data: {materials_with_mp_data} ({materials_with_mp_data/total_original*100:.1f}%)")
        
        # Check for missing data
        new_columns = [col for col in mp_df.columns if col != 'material_id']
        missing_stats = {}
        
        for col in new_columns:
            if col in merged_df.columns:
                missing_count = merged_df[col].isna().sum()
                missing_pct = missing_count / total_merged * 100
                missing_stats[col] = missing_pct
                
                if missing_pct > 50:
                    logger.warning(f"⚠️ {col}: {missing_pct:.1f}% missing data")
                else:
                    logger.info(f"✅ {col}: {missing_pct:.1f}% missing data")
        
        return merged_df
    
    def save_augmented_data(self, 
                           df: pd.DataFrame, 
                           output_path: str,
                           include_metadata: bool = True) -> None:
        """Save augmented dataset with metadata"""
        
        logger.info(f"Saving augmented data to: {output_path}")
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main data
        file_ext = Path(output_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                df.to_csv(output_path, index=False)
            elif file_ext in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            elif file_ext == '.parquet':
                df.to_parquet(output_path, index=False)
            else:
                # Default to CSV
                df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Data saved: {len(df)} materials × {len(df.columns)} features")
            
            # Save metadata
            if include_metadata:
                metadata_path = output_path.replace(file_ext, '_metadata.json')
                
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'total_materials': len(df),
                    'total_features': len(df.columns),
                    'columns': list(df.columns),
                    'missing_data_summary': {
                        col: float(df[col].isna().sum() / len(df) * 100)
                        for col in df.columns
                    },
                    'data_types': {col: str(df[col].dtype) for col in df.columns},
                    'mp_features_added': [
                        col for col in df.columns 
                        if col in ['density', 'volume', 'nsites', 'formation_energy_per_atom',
                                 'energy_above_hull', 'band_gap', 'is_metal', 'num_elements', 
                                 'formula_pretty', 'space_group', 'crystal_system',
                                 'bulk_modulus', 'shear_modulus', 'universal_anisotropy',
                                 'dielectric_total', 'dielectric_ionic', 'dielectric_electronic']
                    ]
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✅ Metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            raise

def main():
    """Main execution function - production ready"""
    
    print("🚀 Production Materials Project Data Augmentation Tool")
    print("="*80)
    
    # Configuration
    MP_API_KEY = os.getenv('MP_API_KEY')
    if not MP_API_KEY:
        print("❌ Please set MP_API_KEY environment variable!")
        print("   Get your API key from: https://next-gen.materialsproject.org/api")
        return
    
    # Get input parameters
    input_file = input("📁 Enter path to existing data file: ").strip()
    if not input_file:
        print("❌ Input file path required!")
        return
    
    output_file = input("💾 Enter output file path (default: augmented_data.csv): ").strip()
    if not output_file:
        output_file = "augmented_data.csv"
    
    rec_file = input("📋 Enter feature recommendations file (optional): ").strip()
    
    batch_size = input("⚙️ Enter batch size (default: 50): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 50
    
    # Initialize fetcher
    fetcher = ProductionMPDataFetcher(MP_API_KEY)
    
    try:
        # Step 1: Read existing data
        logger.info("Step 1: Reading existing data")
        existing_df = fetcher.read_existing_data(input_file)
        
        # Step 2: Identify MP ID column
        logger.info("Step 2: Identifying Materials Project ID column")
        mp_id_column = fetcher.identify_material_id_column(existing_df)
        
        # Step 3: Get feature recommendations
        logger.info("Step 3: Getting feature recommendations")
        features = fetcher.read_feature_recommendations(rec_file)
        
        # Step 4: Validate material IDs
        logger.info("Step 4: Validating material IDs")
        material_ids = existing_df[mp_id_column].dropna().unique().tolist()
        valid_ids = fetcher.validate_material_ids(material_ids)
        
        if not valid_ids:
            logger.error("❌ No valid material IDs found!")
            return
        
        # Step 5: Fetch MP data
        logger.info("Step 5: Fetching Materials Project data")
        mp_df = fetcher.fetch_mp_data_batch(valid_ids, features, batch_size)
        
        if mp_df.empty:
            logger.error("❌ No MP data retrieved!")
            return
        
        # Step 6: Merge datasets
        logger.info("Step 6: Merging datasets")
        merged_df = fetcher.merge_with_existing_data(existing_df, mp_df, mp_id_column)
        
        # Step 7: Save results
        logger.info("Step 7: Saving augmented dataset")
        fetcher.save_augmented_data(merged_df, output_file)
        
        # Final summary
        print("\n🎉 Data augmentation completed successfully!")
        print(f"   📊 Input: {len(existing_df)} materials")
        print(f"   📊 Output: {len(merged_df)} materials with {len(merged_df.columns)} features")
        print(f"   💾 Saved to: {output_file}")
        
        # Show what MP features were added
        mp_features = [col for col in merged_df.columns 
                      if col in ['density', 'volume', 'nsites', 'formation_energy_per_atom',
                               'energy_above_hull', 'band_gap', 'is_metal', 'num_elements', 
                               'formula_pretty', 'space_group', 'crystal_system',
                               'bulk_modulus', 'shear_modulus']]
        print(f"   🔬 MP features added ({len(mp_features)}): {mp_features}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()