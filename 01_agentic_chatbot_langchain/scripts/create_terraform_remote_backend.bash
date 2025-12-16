# Tạo blob storage cho terraform backend để lưu tfstate
# Mục đích của tfstate là lần sau chạy terraform, tf có thể biết so sánh hiện tại và đang có xem có thay đổi không để xử lý.
# Chạy code này trước khi chạy terraform apply phase 1

# 1. Tạo resource group
az group create \
  --name terraform-backend-rg \
  --location southeastasia

# 2. Tạo storage account (tên phải unique)
az storage account create \
  --name tfbackendstorage125 \
  --resource-group terraform-backend-rg \
  --location southeastasia \
  --sku Standard_LRS \
  --encryption-services blob

# 3. Tạo container blob
az storage container create \
  --name tfstate \
  --account-name tfbackendstorage125
