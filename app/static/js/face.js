/* ==========================================================
   人脸识别与视频截取系统 —— 前端逻辑
   依赖 app.js 中定义的全局函数 showToast()
   ========================================================== */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // 系统切换（YOLOv8 / 人脸识别）
  // ------------------------------------------------------------------
  var PAGE_TITLES = {
    yolo: { title: "基于YOLOv8的学生课堂行为分析系统", subtitle: "图片 / 视频 / 本地摄像头 三种检测方式" },
    face: { title: "人脸识别与视频截取系统", subtitle: "视频截取 / 人脸识别 / 添加新人脸" },
  };

  document.querySelectorAll(".nav-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var view = btn.dataset.view;
      document.querySelectorAll(".nav-btn").forEach(function (b) {
        b.classList.toggle("active", b === btn);
      });
      document.getElementById("view-yolo").hidden = view !== "yolo";
      document.getElementById("view-face").hidden = view !== "face";
      var t = PAGE_TITLES[view];
      document.getElementById("page-title").textContent = t.title;
      document.getElementById("page-subtitle").textContent = t.subtitle;
    });
  });

  // ------------------------------------------------------------------
  // 功能切换（视频截取 / 人脸识别 / 添加新人脸）
  // ------------------------------------------------------------------
  function showFaceFunction(value) {
    document.querySelectorAll(".function-config").forEach(function (el) {
      el.hidden = true;
    });
    document.querySelectorAll("#view-face .tab-panel").forEach(function (el) {
      el.classList.remove("active");
    });
    // 防御式处理：部分功能没有独立的侧边栏配置区（如表单在右侧面板）
    var configEl = document.getElementById("face-" + value + "-config");
    if (configEl) configEl.hidden = false;
    document.getElementById("face-panel-" + value).classList.add("active");
  }

  document.querySelectorAll('input[name="face-function"]').forEach(function (radio) {
    radio.addEventListener("change", function () {
      if (radio.checked) showFaceFunction(radio.value);
    });
  });

  // 置信度阈值滑块联动
  var thSlider = document.getElementById("face-threshold");
  var thOut = document.getElementById("face-threshold-out");
  thSlider.addEventListener("input", function () {
    thOut.textContent = (parseInt(thSlider.value, 10) / 100).toFixed(2);
  });

  // 页面加载时检查系统状态（dlib / 模型 / 数据库），给出友好提示
  fetch("/api/face/status")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data && data.message) {
        document.getElementById("face-recognize-info").textContent = data.message;
      }
    })
    .catch(function () { /* 忽略状态检查失败 */ });

  // ------------------------------------------------------------------
  // 功能一：视频截取
  // ------------------------------------------------------------------
  var extractFile = document.getElementById("face-extract-file");
  var extractInfo = document.getElementById("face-extract-info");
  var extractBtn = document.getElementById("face-extract-btn");

  // 选择视频后读取时长信息（用于提示结束时间上限）
  extractFile.addEventListener("change", function () {
    var file = extractFile.files[0];
    if (!file) return;
    var url = URL.createObjectURL(file);
    var video = document.createElement("video");
    video.preload = "metadata";
    video.src = url;
    video.onloadedmetadata = function () {
      var dur = video.duration || 0;
      var endInput = document.getElementById("face-extract-end");
      endInput.max = Math.floor(dur);
      extractInfo.textContent = "时长: " + dur.toFixed(2) + " 秒，结束时间请不超过 " + dur.toFixed(2);
      URL.revokeObjectURL(url);
    };
  });

  extractBtn.addEventListener("click", async function () {
    var file = extractFile.files[0];
    if (!file) {
      showToast("请先选择视频", "info");
      return;
    }
    var formData = new FormData();
    formData.append("file", file);
    formData.append("start_time", document.getElementById("face-extract-start").value);
    formData.append("end_time", document.getElementById("face-extract-end").value);
    formData.append("num_frames", document.getElementById("face-extract-num").value);

    extractBtn.disabled = true;
    extractBtn.textContent = "处理中...";
    try {
      var res = await fetch("/api/face/video/extract", { method: "POST", body: formData });
      var data = await res.json();
      if (!res.ok || data.code !== 0) throw new Error(data.detail || "视频处理失败");

      var grid = document.getElementById("face-extract-grid");
      grid.innerHTML = "";
      data.frame_urls.forEach(function (u, i) {
        var fig = document.createElement("figure");
        fig.className = "frame-card";
        var img = document.createElement("img");
        img.src = u;
        img.alt = "帧 " + (i + 1);
        var cap = document.createElement("figcaption");
        cap.textContent = "帧 " + (i + 1);
        fig.appendChild(img);
        fig.appendChild(cap);
        grid.appendChild(fig);
      });

      extractInfo.textContent =
        "视频信息: 时长 " + data.duration + " 秒 / " + data.fps +
        " FPS / 共 " + data.total_frames + " 帧。成功截取 " + data.extracted + " 张图片";
      showToast("成功截取 " + data.extracted + " 张图片", "success");
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      extractBtn.disabled = false;
      extractBtn.textContent = "截取图片";
    }
  });

  // ------------------------------------------------------------------
  // 功能二：人脸识别
  // ------------------------------------------------------------------
  var recogFiles = document.getElementById("face-recognize-files");
  var recogBtn = document.getElementById("face-recognize-btn");
  var recogInfo = document.getElementById("face-recognize-info");

  recogBtn.addEventListener("click", async function () {
    var files = recogFiles.files;
    if (!files.length) {
      showToast("请先选择图片", "info");
      return;
    }
    var formData = new FormData();
    Array.prototype.forEach.call(files, function (f) {
      formData.append("files", f);
    });
    formData.append("confidence", (parseInt(thSlider.value, 10) / 100).toFixed(2));

    recogBtn.disabled = true;
    recogBtn.textContent = "识别中...";
    try {
      var res = await fetch("/api/face/recognize", { method: "POST", body: formData });
      var data = await res.json();
      if (!res.ok || data.code !== 0) throw new Error(data.detail || "人脸识别失败");

      var container = document.getElementById("face-recognize-results");
      container.innerHTML = "";
      recogInfo.textContent = "数据库共 " + data.total_students_count + " 名学生参与考勤比对";

      data.images.forEach(function (item) {
        var card = document.createElement("div");
        card.className = "face-result";

        var head = document.createElement("h3");
        head.textContent = item.filename || "图片";

        var body = document.createElement("div");
        if (item.error) {
          var errP = document.createElement("p");
          errP.className = "status";
          errP.textContent = item.error;
          body.appendChild(errP);
        } else {
          var compare = document.createElement("div");
          compare.className = "img-compare";
          compare.innerHTML =
            '<figure><img src="data:image/jpeg;base64,' + item.original_b64 +
            '" alt="原始图片"><figcaption>原始图片</figcaption></figure>' +
            '<figure><img src="data:image/jpeg;base64,' + item.annotated_b64 +
            '" alt="识别结果"><figcaption>识别结果</figcaption></figure>';

          var known = item.known_count;
          var unknown = item.unknown_count;
          var present = item.recognized_students.length;
          var absent = item.absent_students.length;

          var stats = document.createElement("div");
          stats.className = "stats";
          stats.innerHTML =
            '<div class="stat-card"><div class="num">' + known + '</div><div class="label">已识别</div></div>' +
            '<div class="stat-card"><div class="num red">' + unknown + '</div><div class="label">未识别</div></div>' +
            '<div class="stat-card"><div class="num">' + (known + unknown) + '</div><div class="label">检测到人脸</div></div>' +
            '<div class="stat-card"><div class="num green">' + present + '</div><div class="label">到场</div></div>' +
            '<div class="stat-card"><div class="num ' + (absent > 0 ? "red" : "green") + '">' + absent + '</div><div class="label">缺席</div></div>';

          var detail = document.createElement("div");
          detail.className = "face-detail";
          if (item.people_info.length) {
            var h4a = document.createElement("h4");
            h4a.textContent = "识别到的人员";
            detail.appendChild(h4a);
            var ulA = document.createElement("ul");
            item.people_info.forEach(function (p) {
              var li = document.createElement("li");
              li.textContent = p.name + "（相似度 " + p.similarity + "）";
              ulA.appendChild(li);
            });
            detail.appendChild(ulA);
          }
          if (item.absent_students.length) {
            var h4b = document.createElement("h4");
            h4b.textContent = "缺席学生";
            detail.appendChild(h4b);
            var ulB = document.createElement("ul");
            item.absent_students.forEach(function (s) {
              var li = document.createElement("li");
              li.textContent = s;
              ulB.appendChild(li);
            });
            detail.appendChild(ulB);
          }

          body.appendChild(compare);
          body.appendChild(stats);
          body.appendChild(detail);
        }

        card.appendChild(head);
        card.appendChild(body);
        container.appendChild(card);
      });

      showToast("识别完成", "success");
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      recogBtn.disabled = false;
      recogBtn.textContent = "开始识别";
    }
  });

  // ------------------------------------------------------------------
  // 功能三：添加新人脸
  // ------------------------------------------------------------------
  var addId = document.getElementById("face-add-id");
  var addName = document.getElementById("face-add-name");
  var addClass = document.getElementById("face-add-class");
  var addFile = document.getElementById("face-add-file");
  var addBtn = document.getElementById("face-add-btn");
  var addResult = document.getElementById("face-add-result");

  // 选择图片后本地预览
  addFile.addEventListener("change", function () {
    var file = addFile.files[0];
    var preview = document.getElementById("face-add-preview");
    if (file) {
      var url = URL.createObjectURL(file);
      preview.src = url;
      preview.hidden = false;
    } else {
      preview.hidden = true;
    }
  });

  addBtn.addEventListener("click", async function () {
    var studentId = addId.value.trim();
    var name = addName.value.trim();
    var file = addFile.files[0];
    if (!studentId || !name || !file) {
      showToast("请填写所有带 * 的必填项！", "info");
      return;
    }
    var formData = new FormData();
    formData.append("student_id", studentId);
    formData.append("name", name);
    formData.append("class_name", addClass.value.trim());
    formData.append("file", file);

    addBtn.disabled = true;
    addBtn.textContent = "提交中...";
    try {
      var res = await fetch("/api/face/add", { method: "POST", body: formData });
      var data = await res.json();
      if (!res.ok || data.code !== 0) throw new Error(data.detail || "添加失败");
      addResult.textContent = data.message;
      addResult.style.color = "#16a34a";
      showToast(data.message, "success");
      addId.value = "";
      addName.value = "";
      addClass.value = "";
      addFile.value = "";
      document.getElementById("face-add-preview").hidden = true;
    } catch (e) {
      addResult.textContent = e.message;
      addResult.style.color = "#dc2626";
      showToast(e.message, "error");
    } finally {
      addBtn.disabled = false;
      addBtn.textContent = "提交到数据库";
    }
  });
})();
