#!/usr/bin/env python3
"""
Diagnostic script to check actual CSV data and column values
"""

import pandas as pd
import sys

def load_csv_with_encoding(filepath):
    """Try multiple encodings"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
        except:
            continue
    return pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', engine='python')

print("="*80)
print("LOADING CSV DATA")
print("="*80)

try:
    df = load_csv_with_encoding('Sales Order Exception report 13 and 14 Nov 2025.csv')
    print(f"\n✓ Loaded {len(df):,} records")
    
    print("\n" + "="*80)
    print("COLUMN NAMES")
    print("="*80)
    for idx, col in enumerate(df.columns, 1):
        print(f"{idx}. '{col}'")
    
    print("\n" + "="*80)
    print("SAMPLE DATA (First 3 rows)")
    print("="*80)
    print(df.head(3).to_string())
    
    print("\n" + "="*80)
    print("KEY COLUMN VALUE SAMPLES")
    print("="*80)
    
    if 'Plant' in df.columns:
        print(f"\nPlant values (unique): {df['Plant'].nunique()}")
        print(f"Sample Plants: {df['Plant'].unique()[:10].tolist()}")
    
    if 'Auth Sell Flag Description' in df.columns:
        print(f"\nAuth Sell Flag Description values:")
        print(df['Auth Sell Flag Description'].value_counts())
    
    if 'Material Status Description' in df.columns:
        print(f"\nMaterial Status Description values:")
        print(df['Material Status Description'].value_counts())
    
    if 'Sold-To Party' in df.columns:
        print(f"\nSold-To Party (unique): {df['Sold-To Party'].nunique()}")
        print(f"Sample Sold-To Party: {df['Sold-To Party'].unique()[:5].tolist()}")
    
    if 'Sales Order Number' in df.columns:
        print(f"\nSales Order Number (unique): {df['Sales Order Number'].nunique()}")
    
    if 'Material Descrption' in df.columns:
        print(f"\nMaterial Descrption (unique): {df['Material Descrption'].nunique()}")
    
    if 'Order Quantity Sales Unit' in df.columns:
        print(f"\nOrder Quantity Sales Unit:")
        print(f"  Total sum: {df['Order Quantity Sales Unit'].sum():,.0f}")
        print(f"  Min: {df['Order Quantity Sales Unit'].min()}")
        print(f"  Max: {df['Order Quantity Sales Unit'].max()}")
    
    print("\n" + "="*80)
    print("TEST FILTER: Plant=1007, Auth Sell Flag Description=No")
    print("="*80)
    
    if 'Plant' in df.columns and 'Auth Sell Flag Description' in df.columns:
        # Convert Plant to string for comparison
        df['Plant'] = df['Plant'].astype(str)
        filtered = df[(df['Plant'] == '1007') & (df['Auth Sell Flag Description'] == 'No')]
        print(f"Records found: {len(filtered)}")
        
        if len(filtered) > 0:
            print("\nSample filtered data:")
            print(filtered[['Plant', 'Auth Sell Flag Description', 'Material Descrption', 'Order Quantity Sales Unit']].head(3))
        else:
            print("\n❌ NO DATA FOUND!")
            print("\nChecking what Plant values exist:")
            print(f"  Plants starting with '1': {df[df['Plant'].str.startswith('1', na=False)]['Plant'].unique()[:20].tolist()}")
            print(f"\nChecking Auth Sell Flag values:")
            print(f"  {df['Auth Sell Flag Description'].unique().tolist()}")
    
except FileNotFoundError:
    print("❌ File not found: 'Sales Order Exception report 13 and 14 Nov 2025.csv'")
    print("\nAvailable files in current directory:")
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for f in csv_files:
        print(f"  - {f}")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
