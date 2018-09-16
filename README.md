# Traffic-Detection
Phần mềm phân loại hình ảnh sử dụng Keras và OpenCV trên ngôn ngữ Python để phân loại xem tuyến đường có kẹt xe hay không.


![Screenshot](test2traffic.jpg)



## I. Nghiên cứu và áp dụng công nghệ phân loại hình ảnh (Image-classifier) sử dụng Python và thư viện máy học Keras.

### Các thư viện cần thiết (Dependencies)
  - numpy
  - tensorflow
  - keras
  - opencv
 
### Mạng lưới CNN/ ConvNeural Net và giả thiết xây dựng một CNN để nhận biết kẹt xe.
  CNN là mạng lưới/ thuật toán gồm các neuron và các bước xử lý hình ảnh để máy tính có thể phân loại hình ảnh. Ngoài ra, CNN còn được áp dụng trong các thuật toán nhận biết đồ vật (Object-detection) để phân loại đồ vật.
  CNN được cấu tạo từ nhiều lớp khác nhau, trong đó, 3 lớp cơ bản và cần thiết nhất để hình thành một cấu trúc CNN là:
  
    1. Convolution.
    2. MaxPool.
    3. Fully-connected (Dense).
    
  Từ các lớp trên, ta hình thành được cấu trúc cơ bản nhất.
  
    (INPUT - CONVlayer - MaxPoollayer - Dense - OUTPUT)
    
  INPUT [80x80x1]:
  
  - Là hình ảnh được nhận vào mạng lưới để phân loại. Để phục vụ mục đích nghiên cứu, ta sẽ giới hạn loại dữ liệu nhận vào CNN của chúng ta xuống hai loại: đường kẹt xe và đường không kẹt xe, và độ phân giải của các hình ảnh này sẽ là 80x80, màu trắng đen (nếu biểu diễn trên máy tính, ta chỉ cần 1 mảng ma trận biểu diễn độ sáng tối của ảnh, không cần đến 3 lớp biểu diễn ba màu cơ bản đỏ, lục, lam )  khi nhập vào để hợp với khả năng xử lý trên máy tính thử nghiệm, như vậy, ảnh của chúng ta sẽ được biểu diễn bằng ma trận [80x80x1]  .**Lưu ý rằng các hình ảnh dữ liệu sẽ phần lớn là hình được chụp ở Việt Nam do phần mềm chỉ phục vụ nhận biết kẹt xe ở Việt Nam. Ngoài ra, không có tiêu chí để phân loại như thế nào là kẹt xe, CNN này sẽ hoạt động dựa trên cách chúng ta nhận biết kẹt xe qua các đặc điểm chung (xe nhiều, giao thông đặc, v.v)**
  
  **CONVlayer:**
  
  - Lớp xử lý hình ảnh và xây dựng neuron đầu tiên trong mạng lưới CNN của chúng ta. Convolution là bước tính toán và xuất ra dữ liệu của một phần của ảnh (ROI: Region of image) bằng các kernel hay filter duyệt qua mọi pixel trên ảnh (để hiểu rõ hơn về cách hoạt động của các filter này, các bạn nên ghé thăm trang http://setosa.io/ev/image-kernels/ để thử nghiệm và nghiên cứu output sau khi một hình ảnh được xử lý bởi một filter). Sau đó, hình ảnh xuất ra sẽ là một hình ảnh xử lý đã được làm rõ các đặc tính chính. Điều này sẽ đưa ra x output khác nhau, với x là số filter mà chúng ta chọn. Như vậy, chúng ta sẽ có một ma trận với độ lớn [80x80xX] để biểu diễn hình ảnh mới được xử lý này .
  
  **MaxPool:**
  
  - Lớp pool sẽ thực hiện một bước extract các đặc tính nổi bật nhất của dữ liệu và bỏ đi các dữ liệu không liên quan băng cách tương tự với lớp CONV, chạy một filter qua khắp ảnh, mỗi ROI mà filter đó chạy qua, lấy giá trị lớn nhất trong ROI đó (tương đương với việc chọn các đặc tính nổi bật nhất), như vậy lớp dữ liệu mới của chúng ta sẽ có ma trận [40x40xX]. Nhưng trước khi qua bước này, chúng ta còn phải đưa dữ liệu từ lớp CONV qua một hàm kích hoạt, để tiện, ta sẽ sử dụng hàm RELU (Rectified Linear unit) là hàm kích hoạt đặc trưng cho mọi lớp. Một hàm kích hoạt sẽ dựa vào trọng lượng của các neuron (hay trong trường hợp này là các pixel) để tính toán, nếu phép tính vượt qua một định mức nào đó, dữ liệu này sẽ ảnh hưởng đến kết quả nhận biết hình ảnh cuối cùng, nếu không, neuron này sẽ bị bỏ qua.
  
  **Dense:**
  
  - Là loại lớp cuối cùng trong cấu trúc CNN. Dense, như tên gọi, sẽ kết nối đặc tất cả các neuron ở lớp này với lớp tiếp theo. Làm như thế sẽ tăng sức chứa và độ chính xác của model này tăng mạnh, bởi vì có càng nhiều liên kết giữa các neuron sẽ càng có nhiều cách để miêu tả được dữ liệu nhập vào mạng lưới và đồng nghĩa bước phân loại cuối cùng sẽ chi tiết hơn. Nhưng, các cấu trúc CNN rất thường gặp trường hợp overfitting, do được nhận quá nhiều dữ liệu không liên quan trong hình ảnh có thể ảnh hưởng rất lớn đên khả năng phân loại sau này. Vì thế, chúng ta cần thêm bước Dropout, loại bỏ các liên kết ngẫu nhiên trong mạng lưới để tránh các lỗi không mong muốn.
  
  ### Data:
  
  Muốn xây dựng một mạng lưới CNN nói riêng, hay muốn xây dựng bất kì neural network nào, bước đầu tiên luôn luôn là chuẩn bị dữ liệu. Do CNN là một mạng lưới tự học và tự động phân loại hình ảnh, nên chúng ta phải luyện cho nó cách phân loại bằng cách chọn lọc một cơ sở dữ liệu gồm rất nhiều hình ảnh và đối tượng cần phân loại tự động, gắn mác nó, và cho nó vào mạng lưới dưới dạng input được xử lý, phương pháp này gọi là supervised learning, ta cho vào một lượng input cực lớn được gắn mác, và mạng lưới sẽ cho ta cấu trúc tự học của nó dùng để phân loại hình ảnh. Bộ dữ liệu của chúng ta sẽ chia làm hai phần, phần test và phần train. Phần train sẽ là phần đảm nhiệm phần lớn dữ liệu, đây sẽ là input được gắn mác hình nào là kẹt xe, hình nào là không, để mạng lưới có thể học và phân loại. Sau mỗi lần train xong, mạng lưới sẽ nhận tiếp một bộ dữ liệu nhỏ hơn gọi là phần test, các bức ảnh trong phần này không được gắn mác, nên muốn biết hình nào là kẹt xe, mạng lưới phải sử dụng cấu trúc tự học mà nó đã xây dựng trong phần train để phân loại hình ảnh, từ đó, ta đo độ chính xác của mạng lưới.
  
  Khái quát qua cách chọn dữ liệu, mình đã tiến hành chọn lọc và gắn mác dữ liệu qua các buớc sau đây:
  
  - Bước 1: Download dữ liệu
  
  Ta gặp vấn đề đầu tiên, làm sao để có được lượng lớn cỡ 700-1000 hình ảnh chọn lọc về kẹt xe. Giải pháp của ta là các cơ sở dữ liệu online được tạo ra để phục vụ cho các mục đích nghiên cứu trí tuệ nhân tạo và khoa học dữ liệu lớn. Mình đã chọn ImageNet làm cơ sở để download hình ảnh không kẹt xe (đúng hơn là đường xá). Nhưng do các hình ảnh chuyên biệt về tình hình kẹt xe tại Việt Nam rất khó có thể tìm thấy trong các cơ sở dữ liệu này, mình đã xài Google Image để download hình ảnh kẹt xe với từ khoá " kẹt xe Việt Nam". Như vậy, chúng ta đã có trong tay hơn 1300 hình ảnh để cho mạng lưới train, 100 hình ảnh để test.
  
  - Bước 2: Gắn mác hình ảnh
  
  Vấn đề tiếp theo là làm sao để gắn mác hơn 1400 hình ảnh tổng cộng. Với số lượng lớn như vậy, mình đã viết một chương trình Python dựa trên một forum trên mạng dùng để gắn mác các hình ảnh này. Khi ta download về, lưu ý rằng các hình ảnh kẹt xe phải nằm riêng lẻ với các hình không kẹt xe trong hai folder khác nhau. Như vậy, ta sẽ sử dụng chương trình Labeler.py có trong repository này để gắn mác từng ảnh. Với mỗi ảnh trong folder, ta sẽ gắn mác "traffic." hoặc là "road." trước tên ảnh bằng dòng code sau:
  
  ```
  def get_traindata():
    img_num=1
    for n in os.listdir(mypath):
        path=os.path.join(mypath,n)
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('./train/'+label+str(img_num)+'.jpg',img)
        img_num+=1
  ```
  
  Với thủ tục trên, ta có label là "traffic" hoặc "road", mypath là địa chỉ folder giữ ảnh, img_num là để đếm ảnh và thêm chỉ số đếm vào tên. Như vậy, ta sẽ có:
  
  
  

  
