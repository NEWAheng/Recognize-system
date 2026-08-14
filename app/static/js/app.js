/* ==========================================================
 * 基于YOLOv8的学生课堂行为分析系统 —— 前端逻辑
 * 负责：配置面板初始化、Tab 切换、图片/视频/摄像头检测
 * ========================================================== */

"use strict";

/* ---------------- 工具函数 ---------------- */

/** 显示全局提示消息 */
function showToast(message, type = "info", duration = 3000) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast ${type}`;
  toast.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => (toast.hidden = true), duration);
}

/** 拼接 URL 查询参数 */
function buildQuery(params) {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      qs.append(key, value);
    }
  });
  return qs.toString();
}

/** 读取当前配置面板参数 */
function getCurrentParams() {
  return {
    conf: (parseInt(document.getElementById("conf-slider").value, 10) / 100).toFixed(2),
    iou: (parseInt(document.getElementById("iou-slider").value, 10) / 100).toFixed(2),
    model: document.getElementById("model-select").value || "",
  };
}

/* ---------------- 配置面板初始化 ---------------- */

async function initConfig() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    if (data.code !== 0) throw new Error(data.detail || "加载配置失败");

    const select = document.getElementById("model-select");
    (data.models || []).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
    // 默认优先选择 .pt 模型（PyTorch 推理，无需 onnxruntime，更可靠）
    const ptOption = Array.prototype.find.call(
      select.options,
      (o) => o.value.toLowerCase().endsWith(".pt")
    );
    if (ptOption) select.value = ptOption.value;
    if (!select.options.length) {
      const opt = document.createElement("option");
      opt.textContent = "无可用模型";
      select.appendChild(opt);
    }
    // 应用默认推理参数
    document.getElementById("conf-slider").value = Math.round(data.default_conf * 100);
    document.getElementById("iou-slider").value = Math.round(data.default_iou * 100);
    syncSliderOutput();
  } catch (err) {
    showToast(`配置加载失败：${err.message}`, "error");
  }
}

/** 滑块与输出文本同步 */
function syncSliderOutput() {
  document.getElementById("conf-out").textContent =
    (parseInt(document.getElementById("conf-slider").value, 10) / 100).toFixed(2);
  document.getElementById("iou-out").textContent =
    (parseInt(document.getElementById("iou-slider").value, 10) / 100).toFixed(2);
}

/* ---------------- 检测类型切换（侧边栏单选） ---------------- */

function initDetectType() {
  document.querySelectorAll('input[name="detect-type"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      // 仅切换 YOLO 视图内的面板，避免误伤人脸识别面板
      document.querySelectorAll("#view-yolo .tab-panel").forEach((p) => p.classList.remove("active"));
      document.getElementById(`panel-${radio.value}`).classList.add("active");
    });
  });
}

/* ---------------- 图片检测 ---------------- */

function initImageDetection() {
  const fileInput = document.getElementById("image-file");
  const detectBtn = document.getElementById("image-detect-btn");

  detectBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
      showToast("请先选择一张图片", "info");
      return;
    }

    const params = getCurrentParams();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("conf", params.conf);
    formData.append("iou", params.iou);
    formData.append("model", params.model);

    detectBtn.disabled = true;
    detectBtn.textContent = "检测中...";
    try {
      const res = await fetch("/api/detect/image", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok || data.code !== 0) {
        throw new Error(data.detail || "图片检测失败");
      }

      // 渲染原始图与标注图
      const origin = document.getElementById("image-origin");
      origin.src = URL.createObjectURL(file);
      document.getElementById("image-result").src = `data:image/jpeg;base64,${data.image_base64}`;
      document.getElementById("image-compare").hidden = false;

      // 渲染检测明细表格
      renderDetectTable(data.rows || []);
      // 渲染类别统计表格
      renderCountTable(data.label_counts || {});
      document.getElementById("image-tables").hidden = false;

      showToast(`检测完成，共检出 ${data.total} 个目标`, "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      detectBtn.disabled = false;
      detectBtn.textContent = "开始检测";
    }
  });
}

function renderDetectTable(rows) {
  const tbody = document.querySelector("#detect-table tbody");
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const [index, label, conf, box] = row;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${index}</td>
      <td>${label}</td>
      <td>${conf}</td>
      <td>(${box.x1}, ${box.y1}) → (${box.x2}, ${box.y2})</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCountTable(counts) {
  const tbody = document.querySelector("#count-table tbody");
  tbody.innerHTML = "";
  Object.entries(counts).forEach(([label, count]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td>${count}</td>`;
    tbody.appendChild(tr);
  });
}

/* ---------------- 视频检测 ---------------- */

function initVideoDetection() {
  const fileInput = document.getElementById("video-file");
  const startBtn = document.getElementById("video-start-btn");
  const stopBtn = document.getElementById("video-stop-btn");
  const saveCheck = document.getElementById("save-frames-check");
  const optionsGrid = document.getElementById("frame-options-grid");
  const videoImg = document.getElementById("video-result");
  const status = document.getElementById("video-status");
  let streamUrl = null; // 当前正在播放的推理流地址

  // 勾选“保存帧图片”时显示帧保存参数
  saveCheck.addEventListener("change", () => {
    optionsGrid.hidden = !saveCheck.checked;
  });

  startBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
      showToast("请先选择一个 mp4 视频", "info");
      return;
    }

    status.textContent = "正在上传视频...";
    startBtn.disabled = true;
    try {
      // 1. 上传视频，获取推理流地址
      const formData = new FormData();
      formData.append("file", file);
      const upRes = await fetch("/api/detect/video/upload", { method: "POST", body: formData });
      const upData = await upRes.json();
      if (!upRes.ok || upData.code !== 0) {
        throw new Error(upData.detail || "视频上传失败");
      }

      // 2. 拼接推理流 URL（携带配置参数与帧保存参数）
      const params = getCurrentParams();
      const extra = { conf: params.conf, iou: params.iou, model: params.model };
      if (saveCheck.checked) {
        extra.save_frames = "true";
        extra.interval_minutes = document.getElementById("interval-minutes").value;
        extra.frames_per_minute = document.getElementById("frames-per-minute").value;
        extra.output_folder = document.getElementById("output-folder").value;
        extra.filename_format = document.getElementById("filename-format").value;
      }
      streamUrl = `${upData.stream_url}&${buildQuery(extra)}`;

      // 3. <img> 指向 MJPEG 推理流，实现实时播放
      videoImg.src = streamUrl;
      status.textContent = "推理流已启动，正在实时识别...（处理完成后自动结束）";
      showToast("视频识别已开始", "success");
    } catch (err) {
      status.textContent = "";
      showToast(err.message, "error");
    } finally {
      startBtn.disabled = false;
    }
  });

  stopBtn.addEventListener("click", () => {
    if (streamUrl) {
      videoImg.src = "";
      streamUrl = null;
      status.textContent = "已停止视频识别";
    }
  });
}

/* ---------------- 摄像头检测 ---------------- */

function initCameraDetection() {
  const startBtn = document.getElementById("camera-start-btn");
  const stopBtn = document.getElementById("camera-stop-btn");
  const cameraImg = document.getElementById("camera-result");
  const status = document.getElementById("camera-status");
  let streaming = false;

  startBtn.addEventListener("click", () => {
    const params = getCurrentParams();
    cameraImg.src = `/api/camera/stream?${buildQuery(params)}`;
    streaming = true;
    status.textContent = "摄像头推理流已启动，请允许浏览器访问摄像头";
    showToast("摄像头识别已开启", "success");
  });
  stopBtn.addEventListener("click", async () => {
    if (!streaming) return;
    cameraImg.src = "";
    streaming = false;
    status.textContent = "已停止摄像头识别";
    // 通知后端释放摄像头资源
    try {
      await fetch("/api/camera/stop");
    } catch (_) {
      /* 忽略网络错误 */
    }
  });
}

/* ---------------- 页面初始化 ---------------- */

document.addEventListener("DOMContentLoaded", () => {
  initDetectType();
  initConfig();
  initImageDetection();
  initVideoDetection();
  initCameraDetection();

  // 滑块变化时同步输出
  document.getElementById("conf-slider").addEventListener("input", syncSliderOutput);
  document.getElementById("iou-slider").addEventListener("input", syncSliderOutput);
});
