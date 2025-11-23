import pandas
import os

os.makedirs("01_DataIngestParsing/data/structured_files", exist_ok=True)

data = {
    "product": ["Computer", "Webcam", "Mouse"],
    "category": ["Electronics", "Electronics", "Accessories"],
    "price": [25.00, 35.50, 45.75],
    "quantity": [100, 150, 200],
    "Stock": [50, 70, 99],
    "description": [
        "A high-quality widget for all your widget needs.",
        "An advanced widget with extra features.",
        "A premium widget for professional use."
    ]
}

#save as csv
df = pandas.DataFrame(data)
csv_path = "../data/structured_files/products.csv"
df.to_csv(csv_path, index=False)

#save as excel with multiple sheets
excel_path = "../data/structured_files/products.xlsx"
with pandas.ExcelWriter(excel_path) as writer:
    df.to_excel(writer, sheet_name="Products", index=False)

    # add to another sheet
    summary_data={
        "category": ["Electronics", "Accessories"],
        "total_items": [2, 1],
        "total_values": [ (25.00*100 + 35.50*150), (45.75*200) ]
    }
    summary_df = pandas.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)