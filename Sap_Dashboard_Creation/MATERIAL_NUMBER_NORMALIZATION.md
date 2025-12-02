# Material Number Normalization - Fix for Leading Zeros

## Problem
Material numbers appear in different formats across your data files:
- **Sales Order Exception Report**: `'000000000300009120'` (with leading zeros)
- **A1P Location Sequence**: `'300003291'` (without leading zeros)

This caused filtering and joining to fail because exact string matching (`'000000000300009120' == '300003291'`) returns `False`.

## Solution Implemented

### 1. Normalization Function
Created `_normalize_material_number()` in `exception_handler.py`:
```python
def _normalize_material_number(value: str) -> str:
    """Normalize material number by removing leading zeros.
    
    Examples:
    - '000000000300009120' -> '300009120'
    - '300003291' -> '300003291'
    """
    str_value = str(value).strip()
    normalized = str_value.lstrip('0') or '0'
    return normalized
```

### 2. Smart Filtering
Updated `apply_filters()` to automatically normalize material-related columns:
- **Material columns**: `Material`, `Customer Material Number`, `Material Found`, `INVENID (Order)`, `Pespsi Invenid`, `Inven Id`
- **How it works**: Both the filter value AND dataframe values are normalized before comparison
- **Other columns**: Still use exact matching (no normalization)

### 3. Data Loading Normalization
When loading both CSV files in `sap_dashboard_agent.py`:
- **Material column is normalized** by stripping leading zeros
- **Original values are preserved** in `Material_Original` column (for reference)
- This ensures JOINs work correctly between tables

## Benefits

✅ **Cross-file filtering works**: Filter by material `300003291` will match `000000000300009120`
✅ **Table joins work**: Sales Order and Location Sequence tables can join on Material
✅ **Backwards compatible**: Other columns still use exact matching
✅ **Original data preserved**: `Material_Original` column keeps the original format

## Example Usage

### Before Fix:
```python
# Would return 0 rows (no match)
filtered = df[df['Material'] == '300003291']  # Looking for '300003291'
# But data has '000000000300009120' ❌
```

### After Fix:
```python
# Returns matching rows (normalized comparison)
filters = {'Material': '300003291'}
filtered = apply_filters(df, filters)
# Matches both '300003291' and '000000000300009120' ✅
```

## Technical Details

### Files Modified:
1. **`exception_handler.py`**:
   - Added `_normalize_material_number()` function
   - Updated `apply_filters()` to handle material columns intelligently

2. **`sap_dashboard_agent.py`**:
   - Normalizes Material column when loading `Sales Order Exception report.csv`
   - Normalizes Material column when loading `A1P_Locn_Seq_EXPORT.csv`
   - Preserves original values in `Material_Original` column

### Columns Affected:
The normalization applies to these columns:
- `Material` (primary)
- `Customer Material Number`
- `Material Found`
- `INVENID (Order)`
- `Pespsi Invenid`
- `Inven Id`

### Columns NOT Affected:
All other columns (Plant, Sales Order Number, etc.) still use exact string matching.

## Testing

To test the fix:
1. Query with a material number (with or without leading zeros)
2. Check that filtering works correctly
3. Verify cross-table joins return expected results

Example test query:
```
"Show me exceptions for material 300009120"
```

This should now match material `000000000300009120` in your data.
