import os
import pandas as pd
import pyreadstat as sp
from tkinter import filedialog
from fastparquet import write 
from tkinter import messagebox as msg
import PySimpleGUI as sg
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import Menu
from PIL import Image, ImageTk
import subprocess
import os
import csv


list = []

def generate_comparison_report(matched_instances, mismatched, secmissmatched, dfs1, dfs2, fivarLabel, sevarLabel, fivalLabel, sevalLabel, flag, radio, parquet_file1, parquet_file2):
    total_iterations = len(matched_instances)
    Totalvariable = len(matched_instances)
    totaltext = sg.Text("Total variables...")
    progTotalvariable = sg.Text(Totalvariable, size=(20, 1))
    progress_bar = sg.ProgressBar(total_iterations, orientation='h', size=(20, 20))
    progress_text = sg.Text('Processing...', size=(20, 1))
    layout = [[totaltext,progTotalvariable],[progress_text], [progress_bar]]
    window = sg.Window('Processing....', layout, finalize=True)
    completed_iterations = 0
    with open(os.path.join(os.path.dirname(parquet_file1), "Comparereport.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Spss variable","Spss variable Label","Precode","Code Labels","File1 Base","File2 Base","File1 Count","File2 Count","File1 Proportion","File2 Proportion","Variance %",f"Flag_{flag}%"])
        for fivar in matched_instances:
            if fivar == "record" : continue
            unique_values = []
            df1_fivar = dfs1[fivar].dropna()
            df2_fivar = dfs2[fivar].dropna()
            unique_values1 = sorted(df1_fivar.unique().tolist())
            unique_values2 = sorted(df2_fivar.unique().tolist())
            unique_values.extend(unique_values1)
            unique_values.extend(unique_values2)
            unique_values = sorted(set(unique_values))
            if len(unique_values) > 100 and radio == False:continue
            for pre in unique_values:
                try:
                    perc1 = len(df1_fivar[df1_fivar == pre]) / len(df1_fivar) * 100 if len(df1_fivar) > 0 else 0
                    count1 = len(df1_fivar[df1_fivar == pre])
                    base1 = len(df1_fivar)
                except:
                    perc1 = 0
                    count1 = 0
                    base1 = 0
                try:
                    perc2 = len(df2_fivar[df2_fivar == pre]) / len(df2_fivar) * 100 if len(df2_fivar) > 0 else 0
                    count2 = len(df2_fivar[df2_fivar == pre])
                    base2 = len(df2_fivar)
                except:
                    perc2 = 0
                    count2 = 0
                    base2 = 0
                perc3 = perc1 - perc2
                flagid = 1 if abs(perc3) >= flag else 0
                try:
                    code_label1 = fivalLabel[fivar].get(pre, "")
                except:
                    code_label1 = ""
                try:
                    code_label2 = sevalLabel[fivar].get(pre, "")
                except:
                    code_label2 = ""
                code_label = code_label1 if code_label1 != "" else code_label2
                writer.writerow([fivar, fivarLabel[fivar], pre, code_label, base1, base2,
                                count1, count2, f"{perc1:.8f}%", f"{perc2:.8f}%",
                                f"{perc3:.8f}%", flagid])
            completed_iterations +=1
            progress_bar.UpdateBar(completed_iterations)
            progress_text.Update(f'Processing: {completed_iterations}/{total_iterations}')
        for fivar in mismatched:
            df1_fivar = dfs1[fivar].dropna()
            unique_values = sorted(df1_fivar.unique().tolist())
            if len(unique_values) > 100 and radio == False:
                continue
            for pre in unique_values:
                perc1 = len(df1_fivar[df1_fivar == pre]) / len(df1_fivar) * 100 if len(df1_fivar) > 0 else 0
                try:
                    code_label1 = fivalLabel[fivar].get(pre, "")
                except:
                    code_label1 = ""
                writer.writerow([fivar, fivarLabel[fivar], pre, code_label1, len(df1_fivar), "", len(df1_fivar[df1_fivar == pre]), "", f"{perc1:.8f}%", "", "", ""])
        for fivar in secmissmatched:
            df2_fivar = dfs2[fivar].dropna()
            unique_values = sorted(df2_fivar.unique().tolist())
            if len(unique_values) > 100 and radio == False:
                continue
            for pre in unique_values:
                perc2 = len(df2_fivar[df2_fivar == pre]) / len(df2_fivar) * 100 if len(df2_fivar) > 0 else 0
                try:
                    code_label2 = sevalLabel[fivar].get(pre, "")
                except:
                    code_label2 = ""
                writer.writerow([fivar, sevarLabel[fivar], pre, code_label2, "", len(df2_fivar), "", len(df2_fivar[df2_fivar == pre]), "", f"{perc2:.8f}%", "", ""])
    os.remove(parquet_file1)
    os.remove(parquet_file2)

def spss():
    file1 = filedialog.askopenfile(title="Please select 1st SPSS file")
    file2 = filedialog.askopenfile(title="Please select 2nd SPSS file")
    sg.theme('lightgreen2')
    layout1 = [[sg.Text('Enter Flag out Percentage',font=('any10bold'))],
            [sg.InputText("", key="Flag", background_color="Beige", size=(30, 1)),sg.Text("Please enter Whole number eg:20")],
            [sg.OK(size=(10,1),button_color='green'), sg.Cancel(size=(10,1))]]
    window = sg.Window('VIVI Formatting', layout1,size= (480,100))
    while True:
        event, values = window.read()
        if event in (None, 'Cancel'):
            break
        elif event == 'OK':
            if values["Flag"] != "":
                flag = int(values["Flag"])
                window.close()
    window.close()
    parquet_file1 = os.path.splitext(file1.name)[0] + '1.gzip.parquet'
    parquet_file2 = os.path.splitext(file2.name)[0] + '2.gzip.parquet'

    def process_spss1(file_path):
        reader = sp.read_file_in_chunks(sp.read_sav, file_path, chunksize=10000)
        first_chunk = True
        qus = []
        metas = []
        for chunk, meta in reader:
            metas.append([meta.variable_value_labels])
            qus.append(meta.column_names_to_labels)
            if first_chunk:
                write(parquet_file1, chunk,compression='gzip')
                first_chunk = False
            else:
                write(parquet_file1, chunk,compression='gzip', append=True)
            valLabel =metas[0][0]
            varLabel = qus[0]
        return valLabel,varLabel
    def process_spss2(file_path):
        reader = sp.read_file_in_chunks(sp.read_sav, file_path, chunksize=10000)
        first_chunk = True
        qus = []
        metas = []
        # parquet_file2 = os.path.splitext(file2)[0] + '.parquet'
        for chunk, meta in reader:
            metas.append([meta.variable_value_labels])
            qus.append(meta.column_names_to_labels)
            if first_chunk:
                write(parquet_file2, chunk,compression='gzip')
                first_chunk = False
            else:
                write(parquet_file2, chunk, compression='gzip',append=True)
            valLabel =metas[0][0]
            varLabel = qus[0]
        return valLabel,varLabel
    fivalLabel, fivarLabel = process_spss1(file1.name)
    print(fivalLabel)
    sevalLabel, sevarLabel = process_spss2(file2.name)
    firstnumeric = []
    secondnumeric = []
    dfs1 = pd.read_parquet(parquet_file1)
    dfs2 = pd.read_parquet(parquet_file2)
    for first in dfs1.columns.tolist():
        if dfs1[first].dropna().replace(".0","").dtype == 'int64' or dfs1[first].dropna().replace(".0","").dtype == 'int32' or dfs1[first].dropna().dtype == "float64":
            firstnumeric.append(first)
    for second in dfs2.columns.tolist():
        if dfs2[second].dropna().replace(".0","").dtype == 'int64' or dfs2[second].dropna().replace(".0","").dtype == 'int32' or dfs2[second].dropna().dtype == "float64":
            secondnumeric.append(second)
    list2_dict = {item: True for item in secondnumeric}
    matched_instances = [item for item in firstnumeric if list2_dict.get(item)]
    del list2_dict
    mismatched = list(set(firstnumeric) - set(secondnumeric))
    secmissmatched = list(set(secondnumeric) - set(firstnumeric))
    # total_iterations = len(matched_instances) + sum(len(dfs1[fivar].dropna().unique().tolist()) for fivar in matched_instances[1:])
    # Totalvariable = len(matched_instances)
    # totaltext = sg.Text("Total variables...")
    # progTotalvariable = sg.Text(Totalvariable, size=(20, 1))
    # progress_bar = sg.ProgressBar(total_iterations, orientation='h', size=(20, 20))
    # progress_text = sg.Text('Processing...', size=(20, 1))
    # layout = [[totaltext,progTotalvariable],[progress_text], [progress_bar]]
    # window = sg.Window('Processing....', layout, finalize=True)
    return matched_instances, mismatched, secmissmatched, dfs1, dfs2, fivarLabel, sevarLabel, fivalLabel, sevalLabel, flag, radio, parquet_file1, parquet_file2

sg.theme('lightgreen2')
layout1 = [#[sg.Text('Please Select which Operation you wish to go for ',font=('any10bold'))],
           [sg.HorizontalSeparator()],
           [sg.HorizontalSeparator()],
           [sg.Button("Compare Two SPSS files",key= "SPSS",enable_events=True,size=(30,2),button_color=('white', 'green'))],
           [sg.HorizontalSeparator()],
           [sg.HorizontalSeparator()],
           [sg.Text("Do you want to get comparision of variable haivng more than 100 precodes?")],
           [sg.Radio("Yes",key = "radio",group_id="yes"),sg.Radio("No",key = "radio",group_id="yes",default=True)],
           [sg.HorizontalSeparator()],
           [sg.HorizontalSeparator()],]
        #    [sg.OK(size=(10,1),button_color='green'), sg.Cancel(size=(10,1))]]
window = sg.Window('Spss Comparision', layout1,size= (550,250))
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    elif event == 'SPSS':
        radio = values["radio"]
        # print(radio)
        matched_instances, mismatched, secmissmatched, dfs1, dfs2, fivarLabel, sevarLabel, fivalLabel, sevalLabel, flag, radio, parquet_file1, parquet_file2 = spss()
        # Call the function
        generate_comparison_report(matched_instances, mismatched, secmissmatched, dfs1, dfs2, fivarLabel, sevarLabel, fivalLabel, sevalLabel, flag, radio, parquet_file1, parquet_file2)
        msg.showinfo(title="Please Check Output Folder",message="Done")
        window.close()
        # msg.showinfo(title="Please Check Output Folder",message="Done")
window.close()

