# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: 
- `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K".
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về **cả bức ảnh** vì trong bản record không có trường tọa độ của một vật thể nào cụ thể, 640x280 chỉ là kích thước của toàn ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Class list là danh sách tất cả các tên class mà model có thể đoán ra. Class list đi kèm taxonomy/checkpoint bởi vì `class_name` được tra ra từ `class_id` trong danh sách của checkpoint mà mỗi `taxonomy_name` đều chỉ ra một name khác nhau trong khi có thể cùng id.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Vì `class_id` và `class_name` chỉ có ý nghĩa trong một `taxonomy_name`. Cùng một số id ở ImageNet và COCO có thể khác nhau. Ghi cả 3 trường giúp người khác tra lại đúng và tái lập được kết quả. 
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Nếu guideline không nói rõ, hai annotator cùng nhìn một ảnh có thể chọn 2 class khác nhau, và nhãn trở nên không nhất quán. Nên cần guideline để có thể sử dụng 1 quy tắc áp dụng cho nhiều ảnh, ví dụ chọn theo tiêu chí nào và làm gì khi 2 vật ngang nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
- Vì `score` là mức độ model tin vào suy đoán của nó, tính từ chính model. Nó không phải do con người kiểm chứng.
- Model có thể tự tin mà vẫn sai, hoặc điểm thấp mà vẫn đúng.
- Ground truth là nhãn do con người gán theo quy tắc. Score chỉ giúp xếp hạng hoặc lọc Prediction, không chứng minh Prediction đúng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: 
- `class_name`="person" ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- góc trên trái ở (385.33, 69.24), góc dưới phải ở (498.92, 348.92), rộng/cao: 113.58/279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
- threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng thì có thể thêm các hộp các điểm thấp vào nên khả năng sai cao hơn, hộp không có vật nào hoặc hộp lệch. Mỗi hộp thêm vào là một thứ **reviewer** phải xem và quyết định giữ hay bỏ. Tăng khối lượng kiểm tra và nguy cơ giữ hộp thừa.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Vẽ một hộp chữ nhật cho mỗi vật thể thuộc danh sách lớp, sao cho các cạnh chạm sát phần nhìn thấy của vật (cao nhất, thấp nhất, trái nhất, phải nhất). Hộp không được cắt vào vật và không chứa khoảng nền thừa đáng kể. Nếu có hai vật cùng loại đứng sát nhau thì vẽ hai hộp riêng, không gộp thành một.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật thể bị che khuất là bowl 0.38. Tôi sẽ đưa lên hỏi Lab Coach bởi vì tôi cảm thấy tình huống này không xuất hiện quá thường xuyên, có thể đặt ra ngưỡng rõ ràng mà không phải đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: 
- `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`= [446.0, 70.0], [445.0, 71.0],[444.0, 71.0].
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon cho thông tin rõ ràng hơn và bám sát hình thật, còn box chỉ là hộp chữ nhật bao quanh vật đó nên vật sẽ chứa nhiều góc thừa, ví dụ con người khi dang tay dang chân.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` dùng để phân biệt từng vật thể riêng trong cùng 1 ảnh. Ví dụ nếu ảnh có 2 người, cả 2 đều là `class_name` = "person" nhưng `instance_id` sẽ là kitchen-001 và kitchen-002. Class ID là id sẽ cho biết loại gì ví dụ person luôn có cùng class_id. Tracking ID dùng để theo dõi cùng một vật thể qua nhiều khung hình trong video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Vẽ biên mask đi sát theo đường viền thật của vật, không cắt vào bên trong vật và không tràn ra nền. Mỗi vật một mask riêng; không gộp hai vật cùng lớp thành một mask. Với phần bị che, chỉ vẽ phần nhìn thấy.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- chỗ 2 vật chồng lên nhau và bị che là bow 0.74 với dinning table 0.58. Tôi nghĩ là sẽ dùng guideline vì phần chồng chất lên nhau này thường xuyên xảy ra đối với polygon, có thể tách được rule khỏi ca cần nói.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn (một lớp trong ImageNet-1K) gẵn cho cả bức ảnh, không có tọa độ. | Ảnh `traffic` có nhiều xe (`cab` 0.51, `minibus` 0.16) nên khó chọn một lớp. | Chọn đúng một lớp trong class list theo guideline (tiêu chí chọn lớp chính, cách xử lý khi hai chủ đề ngang nhau). Không có lớp phù hợp thì escalation, không tự thêm lớp. | Nhãn có thuộc class list không, có là lớp chính theo guideline không, các ảnh nhiều chủ đề có được chọn nhất quán không. Không dựa vào score. |
| Phát hiện vật thể | Mỗi vật một hộp `bbox_xyxy` (pixel) kèm một lớp COCO-80. | Ảnh `kitchen`: `bowl` 0.38 bị che một phần. Số hộp tăng từ 6 (ngưỡng 0.60) lên 17 (ngưỡng 0.20), các hộp điểm thấp có thể thừa hoặc lệch. | Vẽ hộp sát phần nhìn thấy của từng vật, mỗi vật một hộp. Ca bị che chưa có quy tắc thì escalation. | Hộp có sát vật và đúng lớp không, có vật bị sót hoặc hộp thừa không; xem kỹ các hộp điểm thấp. |
| Instance segmentation | Mỗi vật một polygon `polygon_xy` kèm lớp và một instance riêng. | Ảnh `kitchen`: `bowl` 0.74 chồng lên `dining table` 0.58, biên giữa hai vật khó xác định. | Vẽ biên sát viền thật, mỗi vật một mask riêng, phần bị che chỉ vẽ phần nhìn thấy. | Biên có tràn sang vật bên cạnh hoặc cắt vào vật không, hai vật cùng lớp có bị gộp không, chỗ chồng nhau có theo guideline không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Quy tắc bảo vệ dữ liệu của tôi cho bài lab này là chỉ dùng các ảnh công khai mà notebook cung cấp, không tải ảnh cá nhân hay dữ liệu nội bộ lên Colab/Github
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ báo ngay cho Lab Coach vì Lab Coach là người hướng dẫn trực tiếp ở lớp, có thể hỗ trợ tôi ngay trong lúc làm bài

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS
