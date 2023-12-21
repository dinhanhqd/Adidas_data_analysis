import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.express as px
import pygwalker as pyg
import streamlit.components.v1 as components
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="Adidas",
    layout="wide"
)


def load_data(file):
    return pd.read_csv(file)


def show_overview(data):
    st.write("## Đánh Giá Tổng Quan về Dữ Liệu")
    st.write(" Dữ liệu")
    st.write(data)
    st.write(f"Số dòng: {data.shape[0]}")
    st.write("### Thông Tin Về Các Cột")
    st.write(f"Số cột: {data.shape[1]}")
    st.write("Tên các cột:")
    st.write(data.columns)

    columns_of_interest = ['PriceperUnit', 'TotalSales', 'OperatingProfit', 'OperatingMargin']
    # Lọc ra các cột cần quan tâm
    selected_columns = data[columns_of_interest]
    # Tạo DataFrame mới chứa thống kê về giá trị lớn nhất, giá trị nhỏ nhất và giá trị trung bình của các cột cần quan tâm
    selected_stats = selected_columns.describe()
    # Hiển thị thông tin
    st.write("### Thống Kê Các Cột Cụ Thể")
    st.write(selected_stats)


def data_preprocessing(data):
    xoadaudola(data)
    cdNT(data)
    ChuyenPTsangTP(data)
    st.write(data)
    st.write("Thông tin các cột  PriceperUnit, TotalSales, OperatingProfit ")
    st.write(data.describe())

    st.write("Kiểm tra xem dữ liệu có giá trị null không")
    # Kiểm tra giá trị null
    null_values = data.isnull().sum()
    # Hiển thị kết quả
    st.write(null_values)

#-----------------------------------------------------------------------------
def ChuyenPTsangTP(data):
    # Kiểm tra xem cột OperatingMargin có giá trị nào đó chưa
    if 'OperatingMargin' in data.columns:
        # Loại bỏ ký tự % và chuyển đổi sang số thực
        data['OperatingMargin'] = data['OperatingMargin'].replace('%', '', regex=True).astype(float)

        # Chia cho 100 để đưa về định dạng phần trăm
        data['OperatingMargin'] /= 100


def xoadaudola(data):
    # Xử lý cột PriceperUnit
    data['PriceperUnit'] = data['PriceperUnit'].replace('[\$,]', '', regex=True).astype(float)

    # Xử lý cột TotalSales
    data['TotalSales'] = data['TotalSales'].replace('[\$,]', '', regex=True).astype(float)

    # Xử lý cột OperatingProfit
    data['OperatingProfit'] = data['OperatingProfit'].replace('[\$,]', '', regex=True).astype(float)

    return data
def cdNT(data):
    # Chuyển đổi cột ngày về kiểu datetime nếu chưa
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d/%m/%Y')
    # Tạo cột mới để lấy năm từ cột ngày
    data['Year'] = data['InvoiceDate'].dt.year
    # Chuyển đổi kiểu dữ liệu cột 'Year' thành kiểu số nguyên
    data['Year'] = data['Year'].astype(str)
def dttn(data):
    xoadaudola(data)
    col1, col2 = st.columns(2)
    with col1:
        cdNT(data)
        st.write("### Tổng doanh thu theo từng năm")
        # Tính tổng doanh thu theo từng năm
        yearly_total_sales = data.groupby('Year')['TotalSales'].sum().reset_index()
        # Sắp xếp theo doanh thu giảm dần
        yearly_total_sales = yearly_total_sales.sort_values(by='TotalSales', ascending=False)
        # Hiển thị bảng
        st.write(yearly_total_sales)
    with col2:
        fig = px.pie(yearly_total_sales, values='TotalSales', names='Year')
        st.plotly_chart(fig)
def dttt(data):
    xoadaudola(data)
    col1, col2 = st.columns(2)
    with col1:
        cdNT(data)
        st.write("### Tổng Doanh Thu Theo Từng Tháng Của Từng Năm")
        # Tạo cột mới để lấy tháng từ cột ngày
        data['Month'] = data['InvoiceDate'].dt.month
        # Tính tổng doanh thu theo từng năm và từng tháng
        monthly_revenue = data.groupby(['Year', 'Month'])['TotalSales'].sum().reset_index()
        # Hiển thị bảng
        st.write(monthly_revenue)
    with col2:
        # Vẽ biểu đồ cột
        fig = px.bar(monthly_revenue, x='Month', y='TotalSales', color='Year', barmode='group',
                     labels={'TotalSales': 'Doanh Thu', 'Month': 'Tháng'})
        st.plotly_chart(fig)
def NamThangDTCN(data):
    cdNT(data)
    xoadaudola(data)
    col1, col2 = st.columns(2)
    with col1:
        # Tạo cột mới để lấy tháng từ cột ngày
        data['Month'] = data['InvoiceDate'].dt.month
        # Tính tổng doanh thu theo từng năm và từng tháng
        monthly_revenue = data.groupby(['Year', 'Month'])['TotalSales'].sum().reset_index()
        st.write("### 5 Tháng Có Doanh Thu Cao Nhất của Tất Cả Các Năm")
        # Sắp xếp toàn bộ dữ liệu theo doanh thu giảm dần
        overall_top_5_months = monthly_revenue.sort_values(by='TotalSales', ascending=False).head(5)
        # Hiển thị bảng
        st.write(overall_top_5_months)
    with col2:
        # Sắp xếp toàn bộ dữ liệu theo doanh thu giảm dần
        overall_top_5_months = monthly_revenue.sort_values(by=['Year', 'TotalSales'], ascending=[True, False]).groupby(
            'Year').head(5)
        # Hiển thị bảng
        st.write(overall_top_5_months)

    # Vẽ biểu đồ cột
    fig = px.bar(overall_top_5_months, x='Month', y='TotalSales', color='Year',
                    labels={'TotalSales': 'Doanh Thu', 'Month': 'Tháng'})
    st.plotly_chart(fig)
def ThiPhanNhaBanLe(data):
    xoadaudola(data)
    cdNT(data)
    col1,col2 = st.columns(2)
    with col1:
        st.write("### Thị phần các nhà bán lẻ")
        # Group by Retailer and sum Operating Profit, then sort in descending order
        top_retailers = data.groupby('Retailer')['OperatingProfit'].sum().sort_values(ascending=False).reset_index()
        # Increase index by 1
        top_retailers.index += 1
        # Hiển thị bảng dữ liệu
        st.write(top_retailers)
    with col2:
        # Group the data by retailer and sum the total sales for each retailer
        retailer_sales = data.groupby('Retailer')['TotalSales'].sum()
        # Calculate the total sales of all retailers
        total_sales = retailer_sales.sum()
        # Calculate the market share of each retailer by dividing their total sales by the total sales of all retailers
        market_share = retailer_sales / total_sales
        # Create a pie chart using plotly
        fig = px.pie(market_share, values=market_share, names=market_share.index)
        # Show the plot on Streamlit
        st.plotly_chart(fig)
def DoanhSoSP(data):
    xoadaudola(data)
    cdNT(data)

    st.write("### Tổng doanh số theo sản phẩm và nhà bán lẻ")
    # Assume df is your DataFrame
    product_sales = data.groupby(['Retailer', 'Product'])['TotalSales'].sum().reset_index()

    # Show the bar chart using Streamlit
    fig = px.bar(product_sales, x='Retailer', y='TotalSales', color='Product')
    st.plotly_chart(fig)
def TopSpBanChay(data):
    xoadaudola(data)
    cdNT(data)
    col1, col2 = st.columns(2)
    with col1:
        # Group by products and apply sum function on Total Sales Column
        top_selling_products = data.groupby('Product')['TotalSales'].sum().reset_index()
        # Increase index by 1
        top_selling_products.index += 1
        # Hiển thị thông tin trong Streamlit
        st.write("### Top Sản Phẩm Bán Chạy Nhất:")
        st.write(top_selling_products)
    with col2:
    # Tạo biểu đồ thanh
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=top_selling_products, y='Product', x='TotalSales', ax=ax)
        ax.set_yticklabels(ax.get_yticklabels())
        ax.set_xlabel('TotalSales')
        ax.set_ylabel('Product Names')
        ax.set_title('Top Sản Phẩm Bán Chạy')

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)
def TPCoDTCaoNhat(data):
    xoadaudola(data)
    cdNT(data)
    col1,col2 = st.columns(2)
    with col1:
        st.write("### Top thành phố có doanh thu cao nhất")
        # Group by City and sum Operating Profit, then sort in descending order and take the top 10
        top_grossing = data.groupby('City')['OperatingProfit'].sum().sort_values(ascending=False).reset_index().head(10)
        # Increase index by 1
        top_grossing.index += 1
        # Hiển thị DataFrame trong Streamlit
        st.write(top_grossing)
    with col2:
        # Biểu đồ tròn
        data = list(top_grossing['OperatingProfit'])
        labels = list(top_grossing['City'])
        colors = sns.color_palette('bright')[0:5]
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%', explode=(0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        st.pyplot(fig)
def TopPPBanChay(data):
    xoadaudola(data)
    cdNT(data)
    col1,col2 = st.columns(2)
    with col1:
        st.write("### Top phương pháp bán chạy nhất")
        # Group by Sales Method and sum Total Sales, then sort in descending order
        top_methods = data.groupby('SalesMethod')['TotalSales'].sum().sort_values(ascending=False).reset_index()
        # Hiển thị bảng dữ liệu
        st.write(top_methods)
    with col2:
        # Tạo biểu đồ Pie
        fig, ax = plt.subplots(figsize=(10, 8))
        # Colors
        colors = sns.color_palette('bright')[0:5]
        # Vẽ biểu đồ Pie
        ax.pie(top_methods['TotalSales'], labels=top_methods['SalesMethod'], colors=colors, autopct='%.0f%%',
               explode=(0.1, 0, 0), shadow=True)
        # Hiển thị biểu đồ
        st.pyplot(fig)

# -----------------------------------------------------------------------------

def data_Mining(data):
    xoadaudola(data)
    dttn(data)
    dttt(data)
    NamThangDTCN(data)

    st.write("### 10 Ngày Có Doanh Thu Cao Nhất của Tất Cả Các Năm")
    # Chuyển đổi cột ngày về kiểu datetime nếu chưa
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d/%m/%Y')
    # Tính tổng doanh thu theo từng ngày
    daily_revenue = data.groupby('InvoiceDate')['TotalSales'].sum().reset_index()
    # Sắp xếp toàn bộ dữ liệu theo doanh thu giảm dần
    overall_top_10_days = daily_revenue.sort_values(by='TotalSales', ascending=False).head(10)
    # Hiển thị bảng
    st.write(overall_top_10_days)
    ThiPhanNhaBanLe(data)
    DoanhSoSP(data)
    TopSpBanChay(data)
    TPCoDTCaoNhat(data)
    TopPPBanChay(data)

def chart(data):
    xoadaudola(data)
    # Assuming 'Retailer' and 'Total Sales' are column names in your DataFrame
    retailer_sales = data.groupby('Retailer')['TotalSales'].sum()
    # Calculate the total sales of all retailers
    total_sales = retailer_sales.sum()
    # Calculate the market share of each retailer by dividing their total sales by the total sales of all retailers
    market_share = retailer_sales / total_sales
    # Create a pie chart using plotly
    fig = px.pie(market_share, values=market_share, names=market_share.index, title='Thị phần của các nhà bán lẻ')
    # Use Streamlit to display the chart
    st.plotly_chart(fig)
def TrucQuanHoa(data):
    # Load your data
    xoadaudola(data)
    cdNT(data)
    ChuyenPTsangTP(data)
    st.title("Trực quan hóa dữ liệu")

    pyg_html = pyg.walk(data, return_html=True)

    components.html(pyg_html, height=1000, scrolling=True)

    return data
def DataNull(data):
    # Xử lý giá trị NaN
    imputer = SimpleImputer(strategy='mean')  # Thay 'mean' bằng chiến lược xử lý bạn muốn
    X_imputed = imputer.fit_transform(data)
    return X_imputed

def chuyen_doi_chuoi_sang_so(data):
    try:
        # Xử lý giá trị trong cột 'UnitsSold', ví dụ: loại bỏ dấu phẩy và chuyển đổi sang kiểu float
        data['UnitsSold'] = data['UnitsSold'].replace('[\$,]', '', regex=True).astype(float)
        data['OperatingProfit'] = data['OperatingProfit'].replace('[\$,]', '', regex=True).astype(float)
        data['OperatingMargin'] = data['OperatingMargin'].replace('[\$,]', '', regex=True).astype(float)
        data['PriceperUnit'] = data['PriceperUnit'].replace('[\$,]', '', regex=True).astype(float)
        return data
    except ValueError:
        print("Không thể chuyển đổi cột 'UnitsSold' thành kiểu số thực.")
        return None
def Linear_Regression(data):
    data = data.copy()
    chuyen_doi_chuoi_sang_so(data)
    # Load your data
    xoadaudola(data)
    cdNT(data)
    ChuyenPTsangTP(data)
    st.write(data)

    X = data[['UnitsSold', 'OperatingProfit', 'OperatingMargin', 'PriceperUnit']]  # features
    y = data['TotalSales']  # target
    st.write(X)
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Tạo và huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán kết quả trên tập kiểm tra
    y_predicted = model.predict(X_test)

    # Hiển thị giá trị thực tế và dự đoán
    result_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_predicted})
    st.write(result_df)

    # Hiển thị điểm số
    score = model.score(X_test, y_test)
    st.write("Điểm số trên tập kiểm tra:", score)

    # Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    ax.set_xlabel('Thực tế')
    ax.set_ylabel('Dự đoán')
    ax.set_title('So sánh giữa giá trị thực tế và dự đoán')
    st.pyplot(fig)
def Random_Forest_Model(data):
    data = data.copy()
    chuyen_doi_chuoi_sang_so(data)
    #preprocess_data(data)
    xoadaudola(data)
    cdNT(data)
    ChuyenPTsangTP(data)
    st.write(data)

    st.write(data.dtypes)
    # Chọn features và target
    X = data[['UnitsSold', 'OperatingProfit', 'OperatingMargin', 'PriceperUnit']]
    y = data['TotalSales']

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Tạo và huấn luyện mô hình Random Forest
    model = RandomForestRegressor(n_estimators=5)
    model.fit(X_train, y_train)

    # Dự đoán kết quả trên tập kiểm tra
    y_predicted = model.predict(X_test)

    # Hiển thị giá trị thực tế và dự đoán
    result_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_predicted})
    st.write(result_df)

    # Hiển thị điểm số
    score = model.score(X_test, y_test)
    st.write("Điểm số trên tập kiểm tra:", score)

    # Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    ax.set_xlabel('Thực tế')
    ax.set_ylabel('Dự đoán')
    ax.set_title('So sánh giữa giá trị thực tế và dự đoán')
    st.pyplot(fig)
def Descion_Tree_Model(data):
    data = data.copy()
    chuyen_doi_chuoi_sang_so(data)
    #preprocess_data(data)
    xoadaudola(data)
    cdNT(data)
    ChuyenPTsangTP(data)
    st.write(data)
    # Chọn features và target
    X = data[['UnitsSold', 'OperatingProfit', 'OperatingMargin', 'PriceperUnit']]
    y = data['TotalSales']

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Tạo và huấn luyện mô hình Decision Tree Regressor
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Dự đoán kết quả trên tập kiểm tra
    y_predicted = model.predict(X_test)

    # Hiển thị giá trị thực tế và dự đoán
    result_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_predicted})
    st.write(result_df)

    # Hiển thị điểm số
    score = model.score(X_test, y_test)
    st.write("Điểm số trên tập kiểm tra:", score)

    # Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    ax.set_xlabel('Thực tế')
    ax.set_ylabel('Dự đoán')
    ax.set_title('So sánh giữa giá trị thực tế và dự đoán')
    st.pyplot(fig)
def Model(data):
    st.write("Model")
    option = st.selectbox(
        'Lựa chọn Model',
        ('Linear Regression Model', 'Random Forest Model', 'Descion Tree Model'))

    if option == "Linear Regression Model":
        Linear_Regression(data)
    if option == "Random Forest Model":
        Random_Forest_Model(data)
    if option =="Descion Tree Model":
        Descion_Tree_Model(data)

def main():
    st.title("Phân tích dữ liệu Bán Hàng Adidas")

    # Thêm thanh bảng chọn bên trái
    sidebar_options = ["Upload Dữ Liệu", "Đánh Giá Tổng Quan","Tiền Xử Lý Dữ Liệu", "Khai thác dữ liệu","Trực quan hóa","Model dự đoán"]
    selected_option = st.sidebar.selectbox("Dữ liệu", sidebar_options)

    # Khởi tạo session_state nếu chưa có
    if 'data' not in st.session_state:
        st.session_state.data = None

    # Xử lý dữ liệu tùy thuộc vào lựa chọn
    if selected_option == "Upload Dữ Liệu":
        uploaded_file = st.file_uploader("Chọn một file CSV", type=["csv"])
        if uploaded_file is not None:
            if st.session_state.data is None:
                st.session_state.data = load_data(uploaded_file)
            else:
                new_data = load_data(uploaded_file)
                st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)

            st.write("Dữ liệu đã được tải lên:")
            st.write(st.session_state.data.head())

    elif selected_option == "Đánh Giá Tổng Quan":
        if st.session_state.data is not None:
            show_overview(st.session_state.data)

    elif selected_option == "Tiền Xử Lý Dữ Liệu":
        if st.session_state.data is not None:
            data_preprocessing(st.session_state.data)

    elif selected_option == "Khai thác dữ liệu":
        if st.session_state.data is not None:
            data_Mining(st.session_state.data)

    elif selected_option == "Trực quan hóa":
        if st.session_state.data is not None:
            TrucQuanHoa(st.session_state.data)

    elif selected_option == "Model dự đoán":
        if st.session_state.data is not None:
            Model(st.session_state.data)

# Thêm checkbox cho việc chọn hiển thị doanh thu theo từng năm
    dtn = st.sidebar.checkbox("Doanh thu theo từng năm")
    if dtn:
        if st.session_state.data is not None:
           dttn(st.session_state.data)
# Thêm checkbox cho việc chọn hiển thị doanh thu theo từng tháng
    dtt = st.sidebar.checkbox("Doanh thu theo tháng")
    if dtt:
        if st.session_state.data is not None:
           dttt(st.session_state.data)
# Thêm checkbox cho việc chọn hiển thị doanh thu 5 tháng cao nhất
    dtNamTCN = st.sidebar.checkbox("Doanh thu 5 tháng cao nhất")
    if dtNamTCN:
        if st.session_state.data is not None:
            NamThangDTCN(st.session_state.data)
# Thêm checkbox cho việc chọn hiển thị Thị phần các nhà bán lẻ
    ThiPhanNhaBanL = st.sidebar.checkbox("Thị phần các nhà bán lẻ")
    if ThiPhanNhaBanL:
        if st.session_state.data is not None:
            ThiPhanNhaBanLe(st.session_state.data)
# Thêm checkbox cho việc chọn hiển thị doanh thu Tổng doanh số theo sản phẩm và nhà bán lẻ
    Doanhsosp = st.sidebar.checkbox("Tổng doanh số theo sản phẩm và nhà bán lẻ")
    if Doanhsosp:
        if st.session_state.data is not None:
            DoanhSoSP(st.session_state.data)
# Thêm checkbox cho việc chọn hiển thị top san phẩm bán chạy
    TopSp = st.sidebar.checkbox("Top sản phẩm bán chạy nhất")
    if TopSp:
        if st.session_state.data is not None:
            TopSpBanChay(st.session_state.data)
# Thêm checkbox cho việc chọn hiển thị top Thành phố bán chạy
    TopTP = st.sidebar.checkbox("Top sản thành phố chạy nhất")
    if TopTP:
        if st.session_state.data is not None:
            TPCoDTCaoNhat(st.session_state.data)
# Thêm checkbox cho việc chọn hiển thị top Thành phố bán chạy
    TopPP = st.sidebar.checkbox("Top sản phương pháp chạy nhất")
    if TopPP:
        if st.session_state.data is not None:
            TopPPBanChay(st.session_state.data)
if __name__ == "__main__":
    main()
