// 답변 입력창에 마이크 버튼을 붙인다.
// 브라우저 내장 음성 인식(Web Speech API)을 써서 API 비용이 들지 않는다.
// Chrome은 음성을 Google 서버에서 글자로 바꾸고, Firefox는 지원하지 않는다.
// localhost나 https 주소에서만 마이크를 쓸 수 있다.
// 주의: st.html은 스크립트 안에 HTML 태그 모양의 문자열이 있으면 스크립트를 실행하지 않는다.
// 아이콘 같은 요소는 문자열 대신 DOM API로 만든다.
(() => {
  if (window.__aiCoachVoice) return;
  window.__aiCoachVoice = true;

  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const MIC_PATH =
    "M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z";
  const ERRORS = {
    "not-allowed": "마이크 권한이 꺼져 있어요. 주소창 왼쪽에서 마이크를 허용해 주세요",
    "service-not-allowed": "이 브라우저에서는 음성 입력을 쓸 수 없어요",
    "audio-capture": "마이크를 찾지 못했어요",
    "no-speech": "말소리가 들리지 않았어요. 다시 눌러 주세요",
    network: "음성 인식 서버에 연결하지 못했어요",
  };

  let recognition = null;
  let listening = false;

  function micIcon() {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", "20");
    svg.setAttribute("height", "20");
    svg.setAttribute("aria-hidden", "true");
    const path = document.createElementNS(ns, "path");
    path.setAttribute("fill", "currentColor");
    path.setAttribute("d", MIC_PATH);
    svg.appendChild(path);
    return svg;
  }

  const chatInput = () => document.querySelector('[data-testid="stChatInput"]');
  const textarea = () => document.querySelector('[data-testid="stChatInputTextArea"]');

  // React가 관리하는 입력창이라 value를 직접 바꾸고 input 이벤트를 보내야 앱에 반영된다
  function setText(value) {
    const area = textarea();
    if (!area) return;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
    setter.call(area, value);
    area.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function showStatus(message) {
    const box = chatInput()?.querySelector(".voice-status");
    if (!box) return;
    box.textContent = message;
    box.hidden = !message;
  }

  function render() {
    const button = chatInput()?.querySelector(".voice-btn");
    if (!button) return;
    const label = listening ? "음성 입력 멈추기" : "말로 답변하기";
    button.classList.toggle("on", listening);
    button.setAttribute("aria-pressed", String(listening));
    button.setAttribute("aria-label", label);
    button.title = label;
  }

  function stop() {
    if (!listening) return;
    listening = false;
    recognition?.stop();
    render();
    showStatus("");
  }

  function start() {
    if (!Recognition) {
      showStatus("이 브라우저는 음성 입력을 지원하지 않아요. Chrome, Edge, Safari에서 써 주세요");
      return;
    }
    const area = textarea();
    if (!area) return;
    // 이미 입력한 글이 있으면 그 뒤에 이어 붙인다
    const prefix = area.value.trim() ? `${area.value.trimEnd()} ` : "";

    recognition = new Recognition();
    recognition.lang = "ko-KR";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.onresult = (event) => {
      if (!listening) return;
      let spoken = "";
      for (const result of event.results) spoken += result[0].transcript;
      setText(prefix + spoken.trim());
    };
    recognition.onerror = (event) => {
      listening = false;
      render();
      showStatus(ERRORS[event.error] || "음성 인식이 멈췄어요. 다시 눌러 주세요");
    };
    recognition.onend = () => {
      if (!listening) return;
      listening = false;
      render();
      showStatus("");
    };

    listening = true;
    recognition.start();
    render();
    showStatus("듣고 있어요. 다 말했으면 마이크를 다시 누르거나 바로 보내세요");
  }

  function mount() {
    const box = chatInput();
    const submit = box?.querySelector('[data-testid="stChatInputSubmitButton"]');
    if (!submit) {
      stop();
      return;
    }
    if (box.querySelector(".voice-btn")) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "voice-btn";
    button.appendChild(micIcon());
    button.addEventListener("click", () => (listening ? stop() : start()));
    const slot = submit.parentElement;
    slot.parentElement.insertBefore(button, slot);

    const status = document.createElement("div");
    status.className = "voice-status";
    status.setAttribute("role", "status");
    status.hidden = true;
    box.appendChild(status);
    render();
  }

  // 직접 타이핑하거나 보내면 받아쓰기를 멈춘다. 보낸 뒤 늦게 온 인식 결과가 입력창을 다시 채우지 않게 한다.
  document.addEventListener(
    "keydown",
    (event) => {
      if (event.target === textarea()) stop();
    },
    true,
  );
  document.addEventListener(
    "click",
    (event) => {
      if (event.target.closest?.('[data-testid="stChatInputSubmitButton"]')) stop();
    },
    true,
  );

  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      mount();
    });
  }).observe(document.body, { childList: true, subtree: true });
  mount();
})();
