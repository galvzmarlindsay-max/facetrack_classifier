import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Attendance Dashboard", page_icon="📊")

st.title("📊 Attendance Dashboard")

name = st.text_input("Enter Student Name")

# Gumamit ng try-except para hindi mag-crash kung wala pang database file
try:
    conn = sqlite3.connect("attendance.db")

    if name:
        # Mas safe na paraan ng pag-query
        query = "SELECT * FROM attendance WHERE name=?"
        df = pd.read_sql_query(query, conn, params=(name,))

        if not df.empty:
            st.success(f"Showing results for: {name}")
            st.dataframe(df) # Mas magandang tignan kaysa st.write(df)

            # Calculation
            total_days = len(df)
            total_present = len(df[df['status'] == 'Present'])
            total_absent = total_days - total_present

            # Display Summary sa Columns para mas professional tignan
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Days", total_days)
            col2.metric("✅ Present", total_present)
            col3.metric("❌ Absent", total_absent)
        else:
            st.warning(f"No records found for '{name}'.")

    conn.close()
except Exception as e:
    st.error(f"Error connecting to database: {e}")