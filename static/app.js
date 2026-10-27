const fileInput = document.querySelector("#file");
const chooseButton = document.querySelector("#choose");
const chooser = document.querySelector("#chooser");
const sampleButton = document.querySelector("#sample");
const checkButton = document.querySelector("#check");
const preview = document.querySelector("#preview");
const statusLine = document.querySelector("#status");
const results = document.querySelector("#results");
const notices = document.querySelector("#notices");

let selectedFile = null;
let previewUrl = null;
let checkToken = 0;

chooseButton.addEventListener("click", () => {
  chooser.hidden = !chooser.hidden;
  statusLine.textContent = chooser.hidden
    ? statusLine.textContent
    : "Pick the dashcam clip, the sample clip, or upload an MP4.";
});

const longClipName = "camera00_second_clip_197_frames.mp4";

document.querySelector("#use-long").addEventListener("click", () => {
  chooser.hidden = true;
  loadNamedClip("/long-clip.mp4", longClipName, askForNotifications());
});

document.querySelector("#use-dashcam").addEventListener("click", () => {
  chooser.hidden = true;
  loadDashcam(true, askForNotifications());
});

document.querySelector("#use-sample").addEventListener("click", () => {
  chooser.hidden = true;
  loadNamedClip("/sample-clip.mp4", "sample_clip.mp4", askForNotifications());
});

fileInput.addEventListener("change", () => {
  const permission = askForNotifications();
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  chooser.hidden = true;
  setFile(file);
  checkSelectedFile(permission);
});

sampleButton.addEventListener("click", () => {
  loadNamedClip("/long-clip.mp4", longClipName, askForNotifications());
});

loadNamedClip("/long-clip.mp4", longClipName);

async function loadDashcam(runCheck, permission) {
  setBusy(true);
  statusLine.textContent = "Loading camera00_test_5_6s.mp4…";
  try {
    const response = await fetch("/dashcam.mp4");
    if (!response.ok) {
      throw new Error("camera00_test_5_6s.mp4 was not found.");
    }
    const blob = await response.blob();
    setFile(new File([blob], "camera00_test_5_6s.mp4", { type: "video/mp4" }));
    if (runCheck) {
      await checkSelectedFile(permission);
      return;
    }
    statusLine.textContent = "camera00_test_5_6s.mp4 is ready.";
  } catch (error) {
    statusLine.textContent = error.message;
  } finally {
    setBusy(false);
    checkButton.disabled = !selectedFile;
  }
}

checkButton.addEventListener("click", () => {
  checkSelectedFile(askForNotifications());
});

async function checkSelectedFile(permission) {
  if (!selectedFile) return;
  const token = ++checkToken;
  const file = selectedFile;
  results.replaceChildren();
  notices.replaceChildren();
  setBusy(true);
  statusLine.textContent = `Checking ${file.name} at 24 frames per second…`;
  try {
    const analysis = await analyzeClip(file);
    if (token !== checkToken) return;
    if (!analysis.signs.length) {
      showEmpty(analysis);
      return;
    }
    statusLine.textContent = "Looking up the recorded action for each sign…";
    const actions = await predictSigns(analysis.signs);
    if (token !== checkToken) return;
    showAlerts(analysis, actions, permission);
  } catch (error) {
    if (token !== checkToken) return;
    statusLine.textContent = error.message || "Could not check that video.";
  } finally {
    if (token === checkToken) {
      setBusy(false);
      checkButton.disabled = !selectedFile;
    }
  }
}

function setFile(file) {
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  checkButton.disabled = false;
  results.replaceChildren();
  statusLine.textContent = `${file.name} is ready.`;
  preview.onloadedmetadata = () => {
    const seconds = preview.duration;
    if (Number.isFinite(seconds) && seconds > 10.05) {
      statusLine.textContent = `${file.name} is ${seconds.toFixed(1)}s. Only the first 10 seconds are checked.`;
    }
  };
}

function setBusy(busy) {
  checkButton.disabled = busy || !selectedFile;
  sampleButton.disabled = busy;
}

async function analyzeClip(file) {
  const body = new FormData();
  body.append("file", file, file.name || "dashcam.mp4");
  const response = await fetch("/analyze", { method: "POST", body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(messageFrom(payload, "Could not read that video."));
  return payload;
}

async function predictSigns(signs) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signs }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(messageFrom(payload, "Could not look up actions."));
  return payload;
}

function showAlerts(analysis, actions, permission) {
  const bySign = new Map((analysis.detections || []).map((item) => [item.sign, item]));
  results.replaceChildren();
  const spoken = [];
  for (const item of actions) {
    const detection = bySign.get(item.sign);
    const card = document.createElement("article");
    card.className = "card";
    const title = document.createElement("h2");
    title.textContent = item.sign;
    const action = document.createElement("p");
    action.className = "action";
    action.textContent = item.action || "No recorded action for this sign.";
    card.append(title, action);
    if (detection) {
      const meta = document.createElement("p");
      meta.className = "meta";
      meta.textContent = `${detection.frames} frames, ${detection.first_seen_s}s–${detection.last_seen_s}s, match ${Math.round(detection.max_confidence * 100)}%.`;
      card.append(meta);
    }
    results.append(card);
    spoken.push(item.action || `${item.sign} has no recorded action.`);
  }
  const rate = analysis.sampled_fps ? `${analysis.sampled_fps} frames per second` : "the clip";
  statusLine.textContent = `Checked ${analysis.frames_sampled} frames (${rate}). ${actions.length} actions sent as notifications.`;
  speak(spoken.join(" "));
  showBanners(actions);
  pushActions(actions, permission);
}

function showEmpty(analysis) {
  results.replaceChildren();
  const note = document.createElement("p");
  note.className = "empty";
  note.textContent = "No sign stayed visible long enough to confirm. Try a clip where the sign is clear for at least part of a second.";
  results.append(note);
  statusLine.textContent = `Checked ${analysis.frames_sampled} frames and did not confirm a sign.`;
}

function messageFrom(payload, fallback) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  return fallback;
}

function showBanners(actions) {
  notices.replaceChildren();
  for (const item of actions) {
    const banner = document.createElement("article");
    banner.className = "notice";
    const title = document.createElement("p");
    title.className = "notice-title";
    title.textContent = String(item.sign || "Sign").replace(/_/g, " ");
    const body = document.createElement("p");
    body.className = "notice-body";
    body.textContent = item.action || "No recorded action for this sign.";
    banner.append(title, body);
    notices.append(banner);
  }
}

async function loadNamedClip(url, filename, permission) {
  setBusy(true);
  statusLine.textContent = `Loading ${filename}…`;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${filename} was not found.`);
    const blob = await response.blob();
    setFile(new File([blob], filename, { type: "video/mp4" }));
    await checkSelectedFile(permission);
  } catch (error) {
    statusLine.textContent = error.message;
  } finally {
    setBusy(false);
    checkButton.disabled = !selectedFile;
  }
}

function askForNotifications() {
  if (!("Notification" in window)) return Promise.resolve(false);
  if (Notification.permission === "granted") return Promise.resolve(true);
  if (Notification.permission !== "default") return Promise.resolve(false);
  return Notification.requestPermission().then((result) => result === "granted");
}

async function pushActions(actions, permission) {
  let allowed = false;
  if (permission) {
    allowed = await permission;
  } else if ("Notification" in window && Notification.permission === "granted") {
    allowed = true;
  }
  if (!allowed) return 0;

  let registration = null;
  if ("serviceWorker" in navigator) {
    const existing = await navigator.serviceWorker.getRegistrations();
    await Promise.all(
      existing
        .filter((item) => item.scope.includes("/static/"))
        .map((item) => item.unregister())
    );
    registration = await navigator.serviceWorker.register("/sw.js");
    await navigator.serviceWorker.ready;
  }
  for (const item of actions) {
    const title = String(item.sign || "Sign").replace(/_/g, " ");
    const body = item.action || "No recorded action for this sign.";
    if (registration) {
      await registration.showNotification(title, {
        body,
        tag: item.sign,
        renotify: true,
        vibrate: [180, 80, 180],
      });
    } else {
      new Notification(title, { body, tag: item.sign });
    }
  }
  return actions.length;
}

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}
