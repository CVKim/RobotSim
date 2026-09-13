#!/usr/bin/env bash
# 헤드리스 rviz 화면을 일정 간격으로 캡처해 GIF 로 만든다 (ffmpeg 없이 ImageMagick 만 쓴다).
#   source scripts/lib_record.sh
#   rec_start :99 /tmp/rec 0.5      # 디스플레이, 프레임 폴더, 간격(초)
#   ... 데모 진행 ...
#   rec_stop  /mnt/e/.../demo.gif 760
# rviz 3D 뷰만 잘라 낸다(왼쪽 패널·툴바 제외) — 파일이 작아지고 볼 것만 남는다.
REC_CROP="${REC_CROP:-1130x910+470+60}"
REC_PID=""
REC_DIR=""

rec_start() {
    local disp="$1" dir="$2" every="${3:-0.5}"
    REC_DIR="$dir"
    mkdir -p "$dir"
    rm -f "$dir"/f_*.png
    (
        k=0
        while :; do
            # 잡을 때 바로 줄인다 — 1130x910 원본을 수백 장 모으면 convert 가 캐시 한계에 걸린다(실제로 겪음)
            import -display "$disp" -window root -crop "$REC_CROP" +repage -resize "${REC_WIDTH:-760}x" \
                   "$(printf '%s/f_%04d.png' "$dir" "$k")" 2>/dev/null || true
            k=$((k + 1))
            sleep "$every"
        done
    ) &
    REC_PID=$!
}

rec_stop() {
    local out="$1" width="${2:-760}" delay="${3:-12}"
    [ -n "$REC_PID" ] && kill "$REC_PID" 2>/dev/null || true
    sleep 1
    local n
    n=$(ls "$REC_DIR"/f_*.png 2>/dev/null | wc -l)
    if [ "${n:-0}" -lt 4 ]; then
        echo "녹화 프레임이 $n 장뿐이라 GIF 를 만들지 않는다"
        return 0
    fi
    mkdir -p "$(dirname "$out")"
    # rviz 가 아직 안 그린 프레임은 거의 검은 화면이라 파일이 몇백 바이트다 — 크기로 걸러 낸다.
    local list
    list=$(for f in "$REC_DIR"/f_*.png; do [ "$(stat -c%s "$f")" -gt 20000 ] && echo "$f"; done)
    if [ -z "$list" ]; then
        echo "녹화된 프레임에 화면이 없다 (rviz 가 뜨기 전에 끝났다)"
        return 0
    fi
    # 너무 많으면 고르게 솎는다 (GIF 용량과 convert 캐시 한계)
    local total keep step
    total=$(echo "$list" | wc -l)
    keep="${REC_MAX_FRAMES:-110}"
    if [ "$total" -gt "$keep" ]; then
        step=$(( (total + keep - 1) / keep ))
        list=$(echo "$list" | awk -v s="$step" 'NR % s == 1')
    fi
    # shellcheck disable=SC2086  (파일 목록을 그대로 넘긴다)
    convert -limit memory 512MiB -limit map 1GiB -delay "$delay" -loop 0 $list \
            -resize "${width}x" -layers Optimize "$out"
    echo "saved $out ($(echo "$list" | wc -l)/$n 프레임 -> $(du -h "$out" | cut -f1))"
}
