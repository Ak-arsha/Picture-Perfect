import streamlit as st
from supabase import create_client, Client
import os

try:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    supabase: Client = create_client(url, key)
    print("Connection successful!")
    
    # Check Auth
    print("Auth client initialized.")

    # List Buckets
    try:
        res = supabase.storage.list_buckets()
        print("Buckets:", res)
    except Exception as e:
        print(f"Error listing buckets: {e}")

    # Check Tables (Select 1 row)
    for table in ["profiles", "photos", "edits"]:
        try:
            res = supabase.table(table).select("*").limit(1).execute()
            print(f"Table '{table}' detected. Data: {res.data}")
        except Exception as e:
            print(f"Error checking table '{table}': {e}")

except Exception as e:
    print(f"Connection failed: {e}")
