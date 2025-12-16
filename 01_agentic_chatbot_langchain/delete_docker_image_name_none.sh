# Xoá tất cả container dừng trước (nếu có tham chiếu đến các image <none>)
docker container prune -f

# Xoá tất cả image <none>
docker rmi $(docker images --filter "dangling=true" -q)

# Kiểm tra lại
docker images
