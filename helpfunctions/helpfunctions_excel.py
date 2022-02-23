"""
Helpfunctions regarding writing to Excel.
"""

import win32com.client


# -------------------------------


def get_excel_range(*args):
    """
    Get excel range given row and column or two tuples of (row, column).
    -----
    (1, 1) -> 'A1'
    ((1, 1), (3, 4)) -> 'A1:C4' 
    """
    if isinstance(args[0], (tuple, list)):
        return ':'.join([''.join([convert_decimal_to_letter(args[0][1]), str(args[0][0])]),
                         ''.join([convert_decimal_to_letter(args[1][1]), str(args[1][0])])])
    else:
        return ''.join([convert_decimal_to_letter(args[1]), str(args[0])])


# -----------


def convert_decimal_to_letter(num_in):
    """
    Analogous to get label of k-th excel column.
    -----
    1 -> 'A'
    28 -> 'AB'
    """
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    let_out = []
    while num_in > 0:
        num_in -= 1
        ind = num_in % 26
        num_in //= 26
        let_out.append(alphabet[ind])
    return ''.join(let_out[::-1])


# ------------


def to_excel_color(rgb):
    """
    Convert color to excel format.
    -----
    (r, g, b) -> excel color code
    """
    c_out = rgb[0] + 256 * rgb[1] + 256 ** 2 * rgb[2]
    return c_out


# ------------


def get_correct_number_of_sheets(wrbk, num_wanted):
    """
    Initialize new WorkBook with given number of Sheets.
    """
    sh_cnt = wrbk.Sheets.Count
    if sh_cnt < num_wanted:
        [wrbk.Sheets.Add() for i in range(num_wanted - sh_cnt)]
    elif sh_cnt > num_wanted:
        [wrbk.Sheets.Item(1).Delete() for i in range(sh_cnt - num_wanted)]

    # ------------


def get_number_format(col_name):
    """
    Set special number format used for particular columns.
    """
    if col_name.startswith(('Rating', 'Game_Rank_Diff', 'Team_Rank_Diff')):
        nf = '0.00'
    elif col_name in ['W_Ratio', 'Opponent_W_Ratio', 'Avg_Point_Diff'] or col_name.startswith('Game_Wght'):
        nf = '0.000'
    else:
        nf = None

    return nf


# ------------


def print_df(df, xl_app, sheet, top_left=(1, 1)):
    """
    Print DataFrame to the excel sheet. Index will be printed too.
    -----
    Input:
        df - pd.DataFrame to print
        xl_app - win32com Excel object
        sheet - win32com WorkSheet object
        top_left - location to print the DataFrame [default (1, 1) -> A1]
    """
    n_rows, n_cols = df.shape
    r, c = top_left
    # header
    sheet.Range(get_excel_range(r, c)).Value = df.index.name
    sheet.Range(get_excel_range((r, c + 1), (r, c + n_cols))).Value = df.columns.tolist()
    sheet.Range(get_excel_range((r, c), (r, c + n_cols))).Font.Bold = True
    sheet.Range(get_excel_range((r, c), (r, c + n_cols))).Interior.Color = to_excel_color((207, 226, 243))
    # data
    sheet.Range(get_excel_range((r + 1, c), (r + n_rows, c))).Value = [[x] for x in df.index.tolist()]
    sheet.Range(get_excel_range((r + 1, c + 1), (r + n_rows, c + n_cols))).Value = [x[1].tolist() for x in
                                                                                    df.fillna('').iterrows()]
    # number format
    for i, col_name in enumerate(df.columns):
        if get_number_format(col_name) is not None:
            sheet.Range(get_excel_range((r + 1, c + 1 + i), (r + n_rows, c + 1 + i))).NumberFormat = get_number_format(
                col_name)
    # filters
    sheet.Select()
    xl_app.ActiveWorkbook.ActiveSheet.Columns('{}:{}'.format(convert_decimal_to_letter(c),
                                                             convert_decimal_to_letter(c + n_cols))).AutoFilter(1)
    # conditional formatting (zebra stripes)
    # sheet.Range(get_excel_range((2, 1), (n_rows + 1, c + n_cols))).FormatConditions.Add(
    #     win32com.client.constants.xlExpression, '',
    #     '=IF(MOD(SUBTOTAL(103,$A$2:$A2),2)=0,"TRUE","FALSE")')
    sheet.Range(get_excel_range((2, 1), (r + n_rows, c + n_cols))).FormatConditions.Add(
        win32com.client.constants.xlExpression, '', '=IF(MOD(ROW(),2)=1,"TRUE","FALSE")')
    sheet.Range(get_excel_range((2, 1), (r + n_rows, c + n_cols))).FormatConditions(1).Interior.Color = to_excel_color(
        (243, 243, 243))
    # freeze panes
    sheet.Range(get_excel_range(2, 1)).Select()
    xl_app.ActiveWindow.FreezePanes = True
    # fit columns
    sheet.Range(get_excel_range((r, c), (r + n_rows, c + n_cols))).Columns.AutoFit()


# ----------


def create_excel_file_from_df_list(filename, df_list, sheet_names=None):
    """
    Print list of DataFrames to Excel file, one per sheet. Sheet names can be 
    optionally also provided. 
    """
    xl_app = win32com.client.gencache.EnsureDispatch('Excel.Application')
    xl_app.DisplayAlerts = False
    wrbk = xl_app.Workbooks.Add()
    df_len = len(df_list)
    get_correct_number_of_sheets(wrbk, df_len)
    for i, df in enumerate(df_list):
        sheet = wrbk.Sheets.Item(i + 1)
        if sheet_names is not None:
            sheet.Name = sheet_names[i]
        print_df(df, xl_app, sheet)
    wrbk.Sheets.Item(1).Select()
    wrbk.SaveAs(filename, 51)
    wrbk.Close()
    xl_app.Quit()
