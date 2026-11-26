// Crimetor AI assistant: chat UI plus "show me" actions performed on the page.
// On assistant.html it binds to the page's chat card; on other pages it adds a floating button.
(() => {
  const HISTORY_KEY = "crimetor-assistant-history";
  const PENDING_KEY = "crimetor-assistant-pending";
  const OPEN_KEY = "crimetor-assistant-open";
  const PAGES = ["index.html", "pricing.html", "assistant.html"];
  const fileName = location.pathname.split("/").pop();
  const currentPage = PAGES.includes(fileName) ? fileName : "index.html";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const GREETING = "Здравствуйте! Я помогу разобраться с сайтом: покажу демо, сравню тарифы и вживую покажу, как оплатить. Спросите что угодно.";
  const SUGGESTIONS = ["Покажи, как оплатить", "Какой тариф выбрать?", "Покажи сценарий пожара", "Работает ли система?"];

  // Where each target lives. Must match TARGETS in assistant.py.
  const TARGETS = {
    "page-demo": { page: "index.html", selector: "#story" },
    "page-pricing": { page: "pricing.html", selector: "#plans-title" },
    "page-assistant": { page: "assistant.html", selector: ".chat-card" },
    "demo-story": { page: "index.html", selector: "#story" },
    "scenario-switch": { page: "index.html", selector: ".scenario-options" },
    "plans": { page: "pricing.html", selector: ".plan-grid" },
    "plan-start": { page: "pricing.html", selector: "[data-select-plan='start']", card: true },
    "plan-campus": { page: "pricing.html", selector: "[data-select-plan='campus']", card: true },
    "plan-network": { page: "pricing.html", selector: "[data-select-plan='network']", card: true },
    "quote-form": { page: "pricing.html", selector: "#quote-form" },
    "pay-button": { page: "pricing.html", selector: "#quote-pay" },
    "download-button": { page: "pricing.html", selector: "#quote-download" }
  };
  const PLAN_TITLES = { start: "Start", campus: "Campus", network: "Network" };

  const storage = {
    get(key, fallback) {
      try { const raw = sessionStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch { return fallback; }
    },
    set(key, value) {
      try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: chat still works on this page */ }
    },
    remove(key) {
      try { sessionStorage.removeItem(key); } catch { /* ignore */ }
    }
  };
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, reduceMotion ? Math.min(ms, 150) : ms));

  // ---------- chat UI ----------
  let ui;
  let history = storage.get(HISTORY_KEY, []);
  let busy = false;

  function buildFloatingUi() {
    const root = document.createElement("div");
    root.className = "aiw-root";
    root.innerHTML = `
      <button class="aiw-launcher" type="button" aria-expanded="false" aria-controls="aiw-panel">
        <span aria-hidden="true">ИИ</span><span class="aiw-sr">Открыть ИИ-помощника</span>
      </button>
      <section class="aiw-panel" id="aiw-panel" role="dialog" aria-label="ИИ-помощник Crimetor" hidden>
        <header class="aiw-head">
          <div><strong>ИИ-помощник</strong><small>DeepSeek · Crimetor</small></div>
          <button class="aiw-icon" type="button" data-aiw-clear title="Очистить чат">Очистить</button>
          <button class="aiw-icon" type="button" data-aiw-close aria-label="Свернуть">×</button>
        </header>
        <div class="aiw-log" role="log" aria-live="polite"></div>
        <div class="aiw-chips"></div>
        <form class="aiw-form">
          <label class="aiw-sr" for="aiw-input">Ваш вопрос</label>
          <input id="aiw-input" type="text" maxlength="1000" autocomplete="off" placeholder="Спросите помощника" required>
          <button type="submit">Отправить</button>
        </form>
        <p class="aiw-status" role="status" aria-live="polite"></p>
      </section>`;
    document.body.append(root);
    const panel = root.querySelector(".aiw-panel");
    const launcher = root.querySelector(".aiw-launcher");
    const setOpen = (open) => {
      panel.hidden = !open;
      launcher.setAttribute("aria-expanded", String(open));
      root.classList.toggle("is-open", open);
      storage.set(OPEN_KEY, open);
      if (open) { ui.input.focus({ preventScroll: true }); ui.log.scrollTop = ui.log.scrollHeight; }
    };
    launcher.addEventListener("click", () => setOpen(panel.hidden));
    root.querySelector("[data-aiw-close]").addEventListener("click", () => setOpen(false));
    root.querySelector("[data-aiw-clear]").addEventListener("click", clearChat);
    document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !panel.hidden) setOpen(false); });
    return {
      log: root.querySelector(".aiw-log"),
      form: root.querySelector(".aiw-form"),
      input: root.querySelector("#aiw-input"),
      status: root.querySelector(".aiw-status"),
      chips: root.querySelector(".aiw-chips"),
      messageClass: "aiw-msg",
      userClass: "aiw-msg aiw-msg--user",
      setOpen,
      isFloating: true,
      restoreOpen: () => setOpen(storage.get(OPEN_KEY, false))
    };
  }

  function bindPageUi() {
    const log = document.getElementById("chat-log");
    const form = document.getElementById("chat-form");
    const chips = document.getElementById("chat-chips");
    const clear = document.getElementById("chat-clear");
    if (clear) clear.addEventListener("click", clearChat);
    return {
      log, form, chips,
      input: document.getElementById("chat-question"),
      status: document.getElementById("chat-status"),
      messageClass: "chat-message",
      userClass: "chat-message chat-message--user",
      setOpen: () => {},
      isFloating: false,
      restoreOpen: () => {}
    };
  }

  function renderMessage(role, text) {
    const item = document.createElement("article");
    item.className = role === "user" ? ui.userClass : ui.messageClass;
    const label = document.createElement("strong");
    label.textContent = role === "user" ? "Вы" : "Помощник";
    const content = document.createElement("span");
    content.textContent = text;
    item.append(label, content);
    ui.log.append(item);
    ui.log.scrollTop = ui.log.scrollHeight;
    return item;
  }

  function renderAll() {
    ui.log.replaceChildren();
    renderMessage("assistant", GREETING);
    history.forEach((m) => renderMessage(m.role, m.content));
    ui.chips.hidden = history.length > 0;
  }

  function clearChat() {
    history = [];
    storage.set(HISTORY_KEY, history);
    renderAll();
    setStatus("");
  }

  function setStatus(text, tone = "") {
    ui.status.textContent = text;
    ui.status.dataset.tone = tone;
  }

  function renderChips() {
    ui.chips.replaceChildren(...SUGGESTIONS.map((text) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "aiw-chip";
      chip.textContent = text;
      chip.addEventListener("click", () => ask(text));
      return chip;
    }));
  }

  async function ask(question) {
    question = question.trim();
    if (!question || busy) return;
    if (location.protocol === "file:") {
      setStatus("Помощник работает только через сервер: запустите server.py в папке landing и откройте http://localhost:8000", "error");
      return;
    }
    busy = true;
    ui.input.value = "";
    ui.chips.hidden = true;
    history.push({ role: "user", content: question.slice(0, 1000) });
    const node = renderMessage("user", question);
    setStatus("Помощник думает...", "info");
    try {
      const response = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history.slice(-12), page: currentPage })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Помощник не ответил. Попробуйте еще раз.");
      const reply = String(data.reply || "").slice(0, 2000) || "Готово.";
      history.push({ role: "assistant", content: reply });
      history = history.slice(-40);
      storage.set(HISTORY_KEY, history);
      renderMessage("assistant", reply);
      setStatus("");
      busy = false;
      if (Array.isArray(data.actions) && data.actions.length) await runActions(data.actions);
    } catch (error) {
      history.pop();
      node.remove();
      ui.input.value = question;
      ui.chips.hidden = history.length > 0;
      setStatus(error instanceof TypeError ? "Сервер недоступен. Проверьте, что server.py запущен." : error.message, "error");
      busy = false;
    }
  }

  // ---------- actions on the page ----------
  function pageOf(action) {
    if (action.type === "payment") return "pricing.html";
    if (action.type === "scenario") return "index.html";
    return TARGETS[action.target] ? TARGETS[action.target].page : currentPage;
  }

  async function runActions(actions) {
    for (let i = 0; i < actions.length; i += 1) {
      const action = actions[i];
      const page = pageOf(action);
      if (page !== currentPage) {
        // Continue on the other page: the widget there picks the rest up from sessionStorage.
        storage.set(PENDING_KEY, actions.slice(i));
        setStatus("Открываю нужную страницу...", "info");
        await wait(500);
        location.href = `./${page}`;
        return;
      }
      if (ui.isFloating && window.innerWidth < 760) ui.setOpen(false);
      if (action.type === "show") await showTarget(action.target, action.hint);
      else if (action.type === "scenario") await showScenario(action.scenario);
      else if (action.type === "payment") await showPayment(action);
    }
  }

  function findTarget(name) {
    const spec = TARGETS[name];
    if (!spec) return null;
    const visible = [...document.querySelectorAll(spec.selector)].find((el) => el.getClientRects().length > 0);
    const element = visible || document.querySelector(spec.selector);
    return element && spec.card ? element.closest(".plan-card") || element : element;
  }

  let hintNode;
  async function spotlight(element, hint, duration = 4200) {
    if (!element) return;
    element.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    await wait(550);
    document.querySelectorAll(".aiw-spotlight").forEach((el) => el.classList.remove("aiw-spotlight"));
    if (hintNode) hintNode.remove();
    element.classList.add("aiw-spotlight");
    if (hint) {
      hintNode = document.createElement("div");
      hintNode.className = "aiw-hint";
      hintNode.textContent = hint;
      document.body.append(hintNode);
      const rect = element.getBoundingClientRect();
      const left = Math.min(Math.max(16, rect.left + window.scrollX), window.scrollX + document.documentElement.clientWidth - hintNode.offsetWidth - 16);
      const above = rect.top > hintNode.offsetHeight + 24;
      hintNode.style.left = `${left}px`;
      hintNode.style.top = `${(above ? rect.top - hintNode.offsetHeight - 14 : rect.bottom + 14) + window.scrollY}px`;
    }
    const node = hintNode;
    setTimeout(() => {
      element.classList.remove("aiw-spotlight");
      if (node) node.remove();
    }, duration);
  }

  async function showTarget(target, hint) {
    await spotlight(findTarget(target), hint);
    await wait(900);
  }

  async function showScenario(scenario) {
    const buttons = [...document.querySelectorAll(`[data-scenario-option='${scenario}']`)];
    const button = buttons.find((el) => el.getClientRects().length > 0) || buttons[0];
    if (!button) return;
    await spotlight(button, scenario === "fire" ? "Переключаю на сценарий «Пожар»" : "Переключаю на сценарий «Вандализм»", 2600);
    await wait(700);
    button.click();
    await wait(600);
    await spotlight(document.getElementById("story"), "Прокрутите вниз, чтобы увидеть сценарий", 3500);
  }

  function setField(field, value) {
    field.value = value;
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
  }

  async function typeInto(field, text) {
    field.focus({ preventScroll: true });
    field.value = "";
    for (const char of text) {
      field.value += char;
      field.dispatchEvent(new Event("input", { bubbles: true }));
      await wait(45);
    }
    field.dispatchEvent(new Event("change", { bubbles: true }));
  }

  async function showPayment({ plan, period, cameras, organization, email }) {
    const form = document.getElementById("quote-form");
    const planSelect = document.getElementById("quote-plan");
    const periodSelect = document.getElementById("quote-period");
    const camerasInput = document.getElementById("quote-cameras");
    const orgInput = form && form.elements.organization;
    const emailInput = form && form.elements.email;
    const pay = document.getElementById("quote-pay");
    if (!form || !planSelect || !periodSelect || !camerasInput || !pay) return;

    await spotlight(findTarget(`plan-${plan}`), `Шаг 1. Тариф ${PLAN_TITLES[plan]}`, 2400);
    await wait(1500);
    await spotlight(planSelect, `Выбираю тариф ${PLAN_TITLES[plan]}`, 2000);
    setField(planSelect, plan);
    await wait(1100);
    await spotlight(periodSelect, period === "year" ? "Шаг 2. Год со скидкой 15%" : "Шаг 2. Оплата за месяц", 2000);
    setField(periodSelect, period);
    await wait(1100);
    const maxCameras = Number(camerasInput.max) || cameras;
    await spotlight(camerasInput, `Шаг 3. Камер: ${Math.min(cameras, maxCameras)}`, 2000);
    setField(camerasInput, String(Math.min(Math.max(1, cameras), maxCameras)));
    await wait(1100);

    for (const [input, value, label] of [[orgInput, organization, "организацию"], [emailInput, email, "рабочую почту"]]) {
      if (!input) continue;
      if (value) {
        await spotlight(input, `Ввожу ${label}`, 2400);
        await typeInto(input, value);
        await wait(600);
      } else if (!input.value.trim()) {
        await spotlight(input, `Шаг 4. Введите ${label}`, 3200);
        await wait(1600);
      }
    }

    await wait(300);
    const ready = form.checkValidity();
    await spotlight(pay, ready
      ? "Готово. Нажмите, чтобы перейти на страницу оплаты Stripe"
      : "Кнопка включится, когда все поля будут заполнены", 7000);
    if (!ready) {
      const firstInvalid = [...form.elements].find((el) => el.willValidate && !el.checkValidity());
      if (firstInvalid) firstInvalid.focus({ preventScroll: true });
    }
  }

  // ---------- start ----------
  function start() {
    ui = document.getElementById("chat-form") ? bindPageUi() : buildFloatingUi();
    renderChips();
    renderAll();
    ui.form.addEventListener("submit", (event) => {
      event.preventDefault();
      ask(ui.input.value);
    });
    ui.restoreOpen();
    const pending = storage.get(PENDING_KEY, null);
    if (Array.isArray(pending) && pending.length) {
      storage.remove(PENDING_KEY);
      // Let the page finish its own setup (pricing form, demo scene) first.
      setTimeout(() => runActions(pending), 700);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
