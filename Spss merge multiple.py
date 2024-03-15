from tkinter import messagebox as msg
import os
import pyreadstat
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import PySimpleGUI as sg
import gc
gc.collect()

sg.theme("lightgreen2")
layout = [[sg.FolderBrowse("Please Select folder.",key="folder",enable_events=True)],
          [sg.Text("Please provide varibale on which we need to merge data")],
          [sg.Input("Please write variable name",key="unique",enable_events=True)],
          [sg.OK(size=(10,1),button_color='green'), sg.Cancel(size=(10,1))],]
window = sg.Window('Spss Merging', layout,size= (480,200))
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    elif event == 'OK':
        uniquevariable = str(values["unique"])
        folder_path = values["folder"]
        # print(uniquevariable)
        window.close()
window.close()
files = []
# Path to the folder containing SPSS files
# folder_path = '/path/to/your/spss/files/'
# def select_folder():
#     root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     folder_path = filedialog.askdirectory()  # Show folder selection dialog
#     return folder_path
# folder_path = select_folder()
# Main function to process files
# Initialize empty dictionaries to store column labels and variable value labels
column_labels = {}
variable_value_labels = {}
# Initialize an empty DataFrame to store merged data
merged_data = pd.DataFrame()
# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.sav'):  # Check if the file is an SPSS file
        file_path = os.path.join(folder_path, file_name)
        try:
            # Try reading the SPSS file with different encodings and force_ascii option
            df, meta = pyreadstat.read_sav(file_path, encoding='utf-8')
        except pyreadstat._readstat_parser.ReadstatError:
            try:
                df, meta = pyreadstat.read_sav(file_path, encoding='latin1')
            except pyreadstat._readstat_parser.ReadstatError:
                df, meta = pyreadstat.read_sav(file_path, force_ascii=True)
        # df, meta = pyreadstat.read_sav(file_path)  # Read the SPSS file
        # Compare column labels
        current_column_labels = meta.column_names_to_labels
        if not column_labels:
            column_labels = current_column_labels
        elif column_labels != current_column_labels:
            for lab in current_column_labels:
                if lab not in column_labels:
                    column_labels[lab] = current_column_labels[lab]
        # Compare variable value labels
        current_variable_value_labels = meta.variable_value_labels
        for variable, labels in current_variable_value_labels.items():
            if variable in variable_value_labels:
                # print()
                if variable_value_labels[variable] != labels:
                    variable_value_labels[variable].update(labels)
                    # print(variable_value_labels[variable])
            else:
                variable_value_labels[variable] = labels
        # Merge the data into the merged_data DataFrame
        merged_data = pd.concat([merged_data, df], ignore_index=True)
merged_rows = []
for record, group in merged_data.groupby(uniquevariable):
    # Concatenate rows in the group, filling NaN values with non-blank values
    merged_row = group.ffill().bfill().iloc[0]
    # Append the merged row to the list of merged rows
    merged_rows.append(merged_row)

merged_data = pd.DataFrame(merged_rows)
# Drop duplicate 'record' values
merged_data.drop_duplicates(subset=[uniquevariable], keep='first', inplace=True, ignore_index=True)

# Export merged data to .sav format
merged_file_path = folder_path+'/merged_data.sav'
merged_file_path1 = folder_path+'/merged_data.xlsx'
merged_data.to_excel(merged_file_path1,index=False)
pyreadstat.write_sav(merged_data, merged_file_path,column_labels=column_labels,variable_value_labels = variable_value_labels)
msg.showinfo(title="Please Check Output Folder",message="Done")