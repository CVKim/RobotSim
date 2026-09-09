# 대차 도킹 한 프레임의 상태. 전에는 같은 내용을 std_msgs/String 안의 JSON 으로 보냈다 —
# 타입을 만들면 필드 이름·단위가 계약이 되고, rviz·rqt·기록(bag)이 필드 단위로 읽는다.
std_msgs/Header header

string mode                 # follow | rotate | backoff | docked | no_measurement
bool docked

# 측정: 대차 기준 좌표에서 본 고리 (u = 좌우, +가 왼쪽 / v = 앞뒤, 음수가 앞 / yaw = 대차 축 대비 회전)
float32 hook_u_mm
float32 rim_v_mm
float32 yaw_deg

# 결합 자리까지 남은 오차 (제어기 자기 측정 기준)
float32 e_u_mm
float32 e_v_mm
float32 e_yaw_deg

# 이번 프레임에 낸 명령
float32 v_mm_s
float32 w_rad_s

int32 frames                # 지금까지 처리한 측정 프레임 수
int32 misses                # 그중 고리를 못 찾은 프레임 수

# 좌표 변환 사슬 검사: 같은 고리를 (1) 인식 노드의 대차 좌표 값과 (2) TF 로 옮긴 값으로 각각 구했을 때의 최대 차이.
# −1 은 아직 재지 못했다는 뜻(TF 가 아직 없거나 검사를 껐다).
float32 tf_check_mm
