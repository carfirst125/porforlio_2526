# Project Prompt Guide

## Tong hop yeu cau ban dau den hien tai

Tai lieu nay tong ket toan bo cac prompt/yeu cau ban da dua ra trong qua trinh xay dung du an `05_customer_transaction`.

---

## 1) Yeu cau khoi tao moi truong va phu thuoc

- Tu notebook `.ipynb` trong `05_customer_transaction`, tao file `requirements.txt` de co the chay thu du an.
- Bo sung dependencies phuc vu backend va dashboard:
  - FastAPI, Uvicorn, Pydantic, pydantic-settings, joblib, pyarrow
  - Streamlit, requests

---

## 2) Yeu cau xay dung he thong phan tich + ML + API trong `app/`

Ban yeu cau thuc hien toan bo trong thu muc `./app`, theo kieu code chuyen nghiep, module/class, de thay doi tham so de dang.

### 2.1 Data understanding va tham chieu notebook

- Doc thong tin dataset tu `about_dataset.md`.
- Tham khao logic trong `customer-segmentation-recommendation-system.ipynb`.

### 2.2 Pipeline phan tich va modeling

- Viet lai toan bo code phan tich du lieu khach hang, segmentation, recommendation.
- Khong hard-code 1 cach duy nhat; xu ly theo huong adaptive, dua tren data profile.
- Luu artifact model de su dung sau:
  - KMeans (va model khac neu can)
  - scaler / pca / metadata

### 2.3 Mo rong insight marketing

Ngoai segmentation + recommendation, bo sung them cac insight score:

- Customer churn score
- Customer lifetime value (CLV)
- Customer net promotion score (NPS) theo huong proxy tu transaction
- Cac insight khac huu ich cho marketing team de chon campaign

### 2.4 API va van hanh cap nhat

- Xay dung RESTful API bang FastAPI de truy xuat insight theo `CustomerID` latency thap.
- API chi read du lieu da tinh san (khong tinh toan nang trong request).
- Du lieu transaction se cap nhat theo thoi gian:
  - Viet job cap nhat insights chay dinh ky (tach khoi API).
  - API chi dung de trich xuat insight cho 1 hoac nhieu customer ID.

---

## 3) Yeu cau tai lieu huong dan

- Tao `userguide.md` trong `05_customer_transaction/app`.
- Noi dung phai ro:
  - Cau truc thu muc app
  - Danh sach file
  - Moi file co class nao, lam gi
  - Lenh chay pipeline de build model/insights
  - Lenh chay API

---

## 4) Yeu cau xu ly loi runtime API

Ban gap loi khi goi:

- `curl http://127.0.0.1:8000/v1/customers/17850`

Yeu cau:
- Kiem tra traceback trong terminal.
- Fix loi serialization (Pydantic/FastAPI) de endpoint tra du lieu thanh cong.

---

## 5) Yeu cau xay dashboard Streamlit

Ban yeu cau tao thu muc `app/streamlit` va xay webapp voi title:

- `Customer Insight Dashboard`

### 5.1 Part 1 - Data Analysis

- Thiet ke dashboard phan tich tong quan transaction data.
- Tham khao hinh/chart trong `.ipynb` va bo sung chart can thiet theo kien thuc chuyen mon.
- Muc tieu: cho thay cac khia canh insight tong the cua dataset giao dich khach hang.

### 5.2 Part 2 - Customer Personalization

- Co o nhap `Customer ID`.
- Bieu do giao dich theo thoi gian cua khach hang do.
- Bang thong tin features cua khach hang.
- Hien thi ket qua inference bang cach goi API.

---

## 6) Yeu cau nang cao dashboard (visual + business story)

Ban tiep tuc yeu cau cac cai tien:

### 6.1 MTD comparison

- Them chart/section so sanh MTD cua thang cuoi dataset voi thang truoc do (cung ngay cat tuong ung).
- Hien thi:
  - Gia tri MTD
  - Tang truong %
  - Bieu do bar/line so sanh

### 6.2 Hien thi value tren chart

- Bar chart va line chart phai hien value tren chart.

### 6.3 Mau sac bar chart

- Dung gradient color cho bar chart:
  - Gia tri lon mau dam
  - Gia tri nho mau nhat dan

### 6.4 Segment presentation

- Trinh bay phan segment thanh 3 nhom khach hang.
- Co mo ta dac tinh tung nhom tren dashboard.

---

## 7) Yeu cau debug dashboard lap lai nhieu vong

Ban da yeu cau kiem tra va fix nhieu van de:

### 7.1 Loi streamlit TypeError

- Loi `_gradient_bar() got an unexpected keyword argument 'text'`.
- Yeu cau fix de app chay on dinh.

### 7.2 Loi bar value hien `NaN`

Ban nhan manh nhieu lan:

- Hai chart:
  - `Top 15 purchased products by quantity`
  - `Top 15 purchased products by revenue`
- Phai hien thi duoc bar value that (khong phai NaN).
- Khong can legend cho 2 chart nay.
- Can sort va ve chart dua tren gia tri so hop le.
- De xuat xu ly gia tri dang string theo pattern:
  - `"label=...|ten san pham|revenue=VALUE"`
  - chi tach `VALUE` de dua vao chart.

### 7.3 Thac mac CLV theo 3 nhom

- Ban dat van de:
  - Tai sao CLV nhom At-Risk/Growth co the cao hon VIP?
- Yeu cau kiem tra:
  - CLV co tinh dung khong?
  - Co tinh nhat quan giua 3 nhom khong?

---

## 8) Tinh chat mong doi cua implementation

Tu cac prompt cua ban, du an can dam bao:

- Code to chuc theo module/class ro rang.
- Co tinh mo rong va de thay doi tham so.
- Tach bach:
  - Offline processing / model refresh
  - Online serving (API read-only latency thap)
- Co tai lieu su dung day du.
- Co dashboard trinh bay du lieu + insight business + ca nhan hoa theo customer.
- Uu tien tinh on dinh runtime va kha nang debug nhanh.

---

## 9) Prompt mau rut gon de tai su dung (copy/paste)

Ban co the dung prompt tong hop sau cho cac lan lap lai du an:

```text
Trong thu muc 05_customer_transaction, hay xay dung hoan chinh he thong customer insights gom:
1) Doc about_dataset.md va tham khao notebook ipynb de viet lai pipeline adaptive cho cleaning, feature engineering, segmentation, recommendation.
2) Luu model/artifacts de tai su dung (clustering model, scaler, pca, metadata, parquet insights).
3) Bo sung insight score cho marketing: churn risk, CLV, promoter/NPS proxy va campaign hints.
4) Xay FastAPI read-only truy xuat insight theo CustomerID (single + batch), latency thap.
5) Viet job refresh dinh ky (tach khoi API) de cap nhat insights khi transaction data thay doi.
6) Viet tai lieu userguide day du ve cau truc app, class/file, lenh chay pipeline/API.
7) Xay Streamlit dashboard title "Customer Insight Dashboard":
   - Part 1: Data Analysis (tong quan giao dich, MTD compare, top products/countries/hours, segment story 3 nhom).
   - Part 2: Customer Personalization (input customer ID, chart lich su giao dich, bang features, API inference JSON).
8) Toan bo bar/line chart can hien value ro rang, top product chart khong co legend, khong de NaN bar value.
9) Kiem tra loi runtime va fix den khi chay on dinh.
```

