import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


def load_nsl_kdd_data():
    # Định nghĩa tên 42 cột chuẩn của bộ dữ liệu NSL-KDD
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_src_port_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target', 'difficulty_level'
    ]

    # Sửa lại đường dẫn này cho khớp chính xác với vị trí file KDDTrain+.txt trong thư mục project của bạn
    train_path = 'data/archive/nsl-kdd/KDDTrain+.txt'

    print(f"Đang đọc dữ liệu từ: {train_path}")
    try:
        df = pd.read_csv(train_path, names=columns)
    except FileNotFoundError:
        # Thử đường dẫn thay thế nếu cấu trúc thư mục khác một chút
        try:
            train_path = 'data/KDDTrain+.txt'
            df = pd.read_csv(train_path, names=columns)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file KDDTrain+.txt! Hãy kiểm tra lại thư mục 'data'.")
            return None

    return df


def main():
    print("=== HỆ THỐNG PHÁT HIỆN TẤN CÔNG MẠNG (ANOMALY DETECTION) ===")

    df = load_nsl_kdd_data()
    if df is None:
        return

    # Lấy 5000 dòng đầu tiên để chạy thử nghiệm nhanh chóng, mượt mà
    df = df.head(5000).copy()
    print(f"Số lượng bản ghi đang xử lý: {len(df)}")

    # 1. Chuyển đổi nhãn: 'normal' là 1 (Bình thường), các loại tấn công khác là -1 (Bất thường)
    df['anomaly_label'] = df['target'].apply(lambda x: 1 if str(x).strip() == 'normal' else -1)
    y_true = df['anomaly_label']

    # 2. Chọn các đặc trưng (features) bao gồm cả dạng số và dạng chữ cần mã hóa
    feature_cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
    X = df[feature_cols].copy()

    # 3. Ép kiểu toàn bộ các cột chữ (categorical) thành dạng số bằng LabelEncoder (Khắc phục triệt để lỗi string to float)
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        # Chuyển đổi giá trị sang string trước để tránh lỗi dữ liệu hỗn hợp NaN/số
        X.loc[:, col] = le.fit_transform(X[col].astype(str))

    X = pd.get_dummies(X)

    # Sau đó mới thực hiện ép kiểu (hoặc để Pandas tự xử lý sau khi đã get_dummies)
    X = X.astype(float)

    # 4. Chuẩn hóa dữ liệu số học
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Huấn luyện mô hình Isolation Forest
    print("\nĐang huấn luyện mô hình Isolation Forest...")

    # Giả định tỷ lệ dữ liệu tấn công/bất thường trong mẫu là khoảng 20% (contamination=0.2)
    model = IsolationForest(contamination=0.2, random_state=42,n_estimators=100)
    model.fit(X_scaled)

    y_pred = model.predict(X_scaled)  # Trả về 1 (bình thường) và -1 (bất thường)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN DỮ LIỆU THỰC TẾ ---")
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(confusion_matrix(y_true, y_pred))
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=['Bất thường/Tấn công (Attack)', 'Bình thường (Normal)']))

    print("\nHoàn tất thành công! Chương trình chạy không còn lỗi.")


if __name__ == '__main__':
    main()
