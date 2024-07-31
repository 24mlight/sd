function disableBtn() {
  $(".submit-btn").prop("disabled", true); //
  $(".submit-btn").css("background-color", "grey");
  $(".submit-btn").css("cursor", "auto");
  $(".submit-btn:hover").css("background-color", "grey");
  disable_btn = true;
  $(".submit-btn").text("generating...");
}
function enableBtn() {
  $(".submit-btn").prop("disabled", false);
  $(".submit-btn").css("background-color", "rgb(12, 137, 246)");
  $(".submit-btn").css("cursor", "pointer");
  $(".submit-btn:hover").css("background-color", "darkblue");
  disable_btn = false;
  $(".submit-btn").text("Generate");
}
function processErr(err) {
  if (err == "need_verify") {
    $("#dialog-confirm").dialog("open");
  } else {
    alert(err);
  }
  return;
}
let task_id = 1;
var timestamp = new Date().getTime();
var timestampInt = parseInt(timestamp / 1000);
function gen_task_id() {
  return timestampInt.toString() + "_" + task_id.toString();
}
function padBase64String(base64String) {
  var padding = "";
  var numPaddingChars = 4 - (base64String.length % 4);
  if (numPaddingChars < 4) {
    for (var i = 0; i < numPaddingChars; i++) {
      padding += "=";
    }
  }
  return base64String + padding;
}
function generate() {
  disableBtn();
  // 获取所有具有 "tab-btn" 类名的元素
  var tabBtns = document.getElementsByClassName("tab-btn");
  var activeTab;
  // 遍历 tabBtns 数组中的每个元素
  console.log(tabBtns.length);
  for (var i = 0; i < tabBtns.length; i++) {
    // 检查当前遍历到的元素是否具有 "active" 类名
    if (tabBtns[i].classList.contains("active")) {
      // 如果找到具有 "active" 类名的元素，则将其存储在 activeTab 变量中并跳出循环
      activeTab = tabBtns[i];
      break;
    }
  }

  if (activeTab.getAttribute("data-tab") === "txt2img") {
    // txt2img 的函数体
    var prompt = $("#prompt").val();
    var sd_model_name = $("#sd_model").val();
    var vae_model_name = $("#vae_file").val();

    $.ajax({
      url: "/txt2img/",
      type: "POST",
      dataType: "json",
      data: JSON.stringify({
        prompt: prompt,
        sd_model: sd_model_name,
        vae_file: vae_model_name,
        task_id: gen_task_id(),
      }),
      success: function (response) {
        // test
        console.log("test:view.txt2img");
        console.log(response["json_resonse"]);

        console.log("generate success, resp: ", response);
        task_id += 1;
        if ("err" in response && response["err"] != "") {
          enableBtn();
          return processErr(response["err"]);
        }
        const image = new Image();

        image.src = "data:image/png;base64," + response["images"][0];

        image.onload = function () {
          $("#image-container").empty().append(image);
          $(image).on("click", function () {
            $.fancybox.open({
              src: this.src,
              type: "image",
            });
          });
        };
        enableBtn();
      },
      error: function (xhr, status, error) {
        console.error("Ajax error：", error);
        task_id += 1;
        enableBtn();
        alert("generate error");
      },
    });
  } else if (activeTab.getAttribute("data-tab") === "img2img") {
    // img2img 的函数体
    var prompt = $("#prompt-2").val();
    var model_name = $("#sd_model").val();
    var imgElement = $("#img-1 img"); // 获取 img-1 内部的 <img> 标签
    var init_images = imgElement.attr("src"); // 获取 <img> 标签的 src 属性
    var imageList = [];
    imageList.push(init_images);

    $.ajax({
      url: "/img2img/",
      type: "POST",
      dataType: "json",
      data: JSON.stringify({
        prompt: prompt,
        init_images: imageList,
        model: model_name,
        task_id: gen_task_id(),
      }),
      success: function (response) {
        console.log("generate success, resp: ", response);
        task_id += 1;
        if ("err" in response && response["err"] != "") {
          enableBtn();
          return processErr(response["err"]);
        }
        const image = new Image();

        image.src = "data:image/png;base64," + response["images"][0];

        image.onload = function () {
          $("#img-2").empty().append(image);
          $(image).on("click", function () {
            $.fancybox.open({
              src: this.src,
              type: "image",
            });
          });
        };
        enableBtn();
      },
      error: function (xhr, status, error) {
        console.error("Ajax error：", error);
        task_id += 1;
        enableBtn();
        alert("generate error");
      },
    });
  }
}

function generate_fallbck() {
  disableBtn();
  var prompt = $("#prompt").val();
  $.ajax({
    url: "/txt2img_fallback/",
    type: "POST",
    dataType: "json",
    data: JSON.stringify({
      prompt: prompt,
      task_id: gen_task_id(),
    }),
    success: function (response) {
      console.log("generate success, resp: ", response);
      task_id += 1;
      if ("err" in response && response["err"] != "") {
        enableBtn();
        return processErr(response["err"]);
      }
      const image = new Image();

      image.src = response["img_data"];

      image.onload = function () {
        $("#image-container").empty().append(image);
        $(image).on("click", function () {
          $.fancybox.open({
            src: image.src,
            type: "image",
          });
        });
      };
      enableBtn();
    },
    error: function (xhr, status, error) {
      console.error("Ajax error：", error);
      task_id += 1;
      enableBtn();
      alert("generate error");
    },
  });
}

function listmodel() {
  $.ajax({
    url: "/list_models/",
    type: "POST",
    dataType: "json",
    data: {},
    success: function (response) {
      console.log("generate success, resp: ", response);
      if ("err" in response && response["err"] != "") {
        return processErr(response["err"]);
      }
      $("#model-name").empty();
      for (let index = 0; index < response["models"].length; index++) {
        $("#model-name").append(response["models"][index] + "<br/>");
      }
    },
    error: function (xhr, status, error) {
      console.error("Ajax error：", error);
      alert("generate error");
    },
  });
}

function progress() {
  if (finished) return;
  $.ajax({
    url: "/progress/",
    type: "POST",
    dataType: "json",
    data: JSON.stringify({
      task_id: gen_task_id(),
    }),
    success: function (response) {
      console.log("generate success, resp: ", response);
      if ("err" in response && response["err"] != "") {
        //   enableBtn();
        return processErr(response["err"]);
      }
      $("#pg")
        .empty()
        .append("progress：" + response["progress"] + "%");
      $("#eta")
        .empty()
        .append("eta(seconds)" + response["eta"]);
    },
    error: function (xhr, status, error) {
      console.error("Ajax error：", error);
      // enableBtn();

      alert("generate error");
    },
  });
}

function openTab(tabName) {
  var i, tabContent, tabBtns;

  // 获取所有具有 "tab" 类名的元素，并将它们存储在变量 tabContent 中
  tabContent = document.getElementsByClassName("tab");
  // 遍历 tabContent 数组中的每个元素
  for (i = 0; i < tabContent.length; i++) {
    // 将当前遍历到的元素的 display 属性设置为 "none"，使其隐藏
    tabContent[i].style.display = "none";
  }

  // 获取所有具有 "tab-btn" 类名的元素，并将它们存储在变量 tabBtns 中
  tabBtns = document.getElementsByClassName("tab-btn");
  // 遍历 tabBtns 数组中的每个元素
  for (i = 0; i < tabBtns.length; i++) {
    // 使用正则表达式替换当前遍历到的元素的类名中的 " active" 为空字符串，移除激活状态
    tabBtns[i].className = tabBtns[i].className.replace(" active", "");
  }

  // 根据传入的 tabName 参数获取对应的元素，并将其 display 属性设置为 "block"，使其显示
  document.getElementById(tabName).style.display = "block";
  // 为触发事件的元素（即当前点击的按钮）添加 " active" 类名，表示激活状态
  event.currentTarget.className += " active";
}
