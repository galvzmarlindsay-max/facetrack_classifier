import sqlite3

def search_student(name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("SELECT date, time, status FROM attendance WHERE name=?", (name,))
    records = cursor.fetchall()

    if not records:
        print("No records found for", name)
        return

    print(f"\nAttendance Record for {name}:")
    print("-" * 40)

    present_count = 0
    absent_count = 0

    for row in records:
        date, time, status = row
        print(f"Date: {date} | Time: {time} | Status: {status}")

        if status == "Present":
            present_count += 1
        else:
            absent_count += 1

    print("\nSummary:")
    print(f"Total Present: {present_count}")
    print(f"Total Absent: {absent_count}")

    conn.close()