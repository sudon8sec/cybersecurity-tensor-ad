import pandas as pd
import numpy as np
import glob
import os
import gc
import boto3

def clean_dataset(df):
    # Remove inconsistent column names with extra spaces
    df.columns = df.columns.str.strip()
    
    # Columns to be removed
    drop_columns = [
        "Destination Port",  # specific targets in simulation
        'Fwd Header Length.1'  # Duplicate column
    ]
    df.drop(columns=drop_columns, inplace=True, errors="ignore")
    
    # Correct columns to the correct dtype
    int_col = df.select_dtypes(include='integer').columns
    df[int_col] = df[int_col].apply(pd.to_numeric, errors='coerce', downcast='integer')
    float_col = df.select_dtypes(include='float').columns
    df[float_col] = df[float_col].apply(pd.to_numeric, errors='coerce', downcast='float')
    
    df['Label'].replace({'BENIGN': 'Benign'}, inplace=True)
    df['Label'] = df.Label.astype('category')
    
    # Remove NaN and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True, subset=df.columns.difference(['Label']))
    
    return df

def main():
    s3_client = boto3.client('s3')
    
    # Specify your bucket name directly or get it from job parameters
    bucket = 'amazon-sagemaker-156804830397-us-east-1-d506qddnjv1mhz'
    
    # List CSV files from S3
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix='cybersecurity-tensor-ad/raw/'
    )
    
    csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files in S3")
    print(f"Using bucket: {bucket}")
    
    all_data = []
    
    for s3_key in csv_files:
        filename = os.path.basename(s3_key)
        print(f"Processing: {filename}")
        
        try:
            # Download and read CSV from S3
            local_path = f'/tmp/{filename}'
            s3_client.download_file(bucket, s3_key, local_path)
            
            df = pd.read_csv(local_path)
            df_clean = clean_dataset(df)
            all_data.append(df_clean)
            
            # Clean up local file
            os.remove(local_path)
            
            del df, df_clean
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            gc.collect()
    
    # Combine all data
    print("Combining all datasets...")
    all_cleaned = pd.concat(all_data, ignore_index=True)
    
    # Split into benign and malicious
    all_benign = all_cleaned[all_cleaned['Label'] == 'Benign'].copy()
    all_malicious = all_cleaned[all_cleaned['Label'] != 'Benign'].copy()
    
    print(f"All cleaned: {all_cleaned.shape}")
    print(f"All benign: {all_benign.shape}")
    print(f"All malicious: {all_malicious.shape}")
    
    # Save to S3
    s3_client = boto3.client('s3')
    
    datasets = {
        'all_cleaned.parquet': all_cleaned,
        'all_benign.parquet': all_benign,
        'all_malicious.parquet': all_malicious
    }
    
    for filename, dataset in datasets.items():
        local_path = f'/tmp/{filename}'
        s3_key = f'cybersecurity-tensor-ad/processed/{filename}'
        
        print(f"Saving {filename}...")
        dataset.to_parquet(local_path, index=False)
        s3_client.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded to s3://{bucket}/{s3_key}")
        
        os.remove(local_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()