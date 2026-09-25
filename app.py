"""
app.py - local web page for the conveyor digital twin.   Run:  python app.py

The page is generated from config.py (panels, plots, controls, divergence rules), so most changes
need no edit here. Edit this file only to change how things LOOK.
"""
import threading
import time
import webbrowser

import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, ctx, dcc, html, no_update
from flask import Response

import config as cfg
import controls
import twin_model
import vision
from mqtt_bridge import MqttBridge
from serial_bridge import SerialBridge
from state import is_num, state

L = cfg.BELT_LENGTH_CM
VAR = {v["id"]: v for v in cfg.VARIABLES}
bridge = SerialBridge()        # conveyor (motor + encoder): Arduino UNO over USB serial
mqtt = MqttBridge()            # camera/vision only: vision.py publishes detections here

app = Dash(__name__, title="Conveyor digital twin", update_title=None)
server = app.server

INK, COBALT, AMBER, PURPLE, GREEN, RED, GREY = "#16212c", "#1f5fbf", "#d98a00", "#7a3fb0", "#2f8f5b", "#c2352b", "#8a97a8"


# ════════════════════════════════════════════════════════════════════════════
# Camera stream for the browser (multipart MJPEG)
# ════════════════════════════════════════════════════════════════════════════
@server.route("/video_feed")
def video_feed():
    def gen():
        last = None
        while True:
            jpg = state.jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ════════════════════════════════════════════════════════════════════════════
# Small render helpers
# ════════════════════════════════════════════════════════════════════════════
def fmt_value(v, val):
    if val is None:
        return "—"
    if is_num(val):
        return v.get("fmt", "{:.1f}").format(val)
    return str(val)


def level(v, val):
    if not is_num(val):
        return "ok"
    for name in ("alarm", "warn"):
        lim = v.get(name)
        if lim:
            lo, hi = lim
            if (lo is not None and val < lo) or (hi is not None and val > hi):
                return name
    return "ok"


def badge(label, text, lvl):
    return html.Span(className=f"badge {lvl}", children=[html.Span(className="dot"), html.B(label), " ", text])


def card(title, *children, cls=""):
    return html.Section(className=f"card {cls}", children=[html.H3(title), *children])


def panels_view():
    placed = {vid for p in cfg.PANELS for vid in p["vars"]}
    panels = list(cfg.PANELS)
    other = [v["id"] for v in cfg.VARIABLES if v["id"] not in placed and not v.get("hidden")]
    if other:
        panels.append(dict(title="Other", vars=other))
    out = []
    for p in panels:
        rows = []
        for vid in p["vars"]:
            v = VAR.get(vid)
            if v is None:
                continue
            val = state.get(vid)
            rows.append(html.Div(className=f"kv {level(v, val) if val is not None else 'stale'}", children=[
                html.Span(v["label"], className="k"),
                html.Span(className="vv", children=[html.Span(fmt_value(v, val), className="v"),
                                                    html.Span(v.get("unit", ""), className="u")]),
            ]))
        out.append(card(p["title"], *rows))
    return out


def divergence_view():
    rows = []
    for r in cfg.DIVERGENCE_RULES:
        st = state.divergences.get(r["id"], {})
        if not st.get("known"):
            lvl, chip = "na", "n/a"
        elif st.get("active"):
            lvl, chip = "alarm", "Alert"
        else:
            lvl, chip = "ok", "OK"
        delta = st.get("delta")
        limit = r.get("abs_tol", r.get("max_age_s"))
        detail = "" if delta is None else f"Δ {delta:.1f} / limit {limit:g} {r.get('unit', '')}"
        body = [html.Div(className="dv-head", children=[
            html.Span(r["label"], className="k"), html.Span(chip, className=f"chip {lvl}")]),
            html.Div(detail, className="dv-detail")]
        if st.get("active"):
            body.append(html.Div(className="dv-why", children=[html.B("Likely cause: "), r["cause"], html.Br(),
                                                                html.B("Suggested action: "), r["action"]]))
        rows.append(html.Div(className=f"dv {lvl}", children=body))
    note = None if state.sync else html.P("Sandbox mode: divergence checks are paused.", className="hint")
    return card("Divergence: physical vs digital", *rows, note)


def belt_objects():
    """Objects to draw/list, from the sensor picked with the "Object position from" control.
    The ultrasonic sensor only sees the object nearest to it, so it gives at most one object."""
    if state.params.get("pos_source", "camera") != "ultrasonic":
        return list(state.objects)
    x = cfg.us_position_cm(state.get("us_distance_cm"))
    if x is None:
        return []
    approach = state.get("us_obj_speed_cm_s")
    speed = None if approach is None else -cfg.ULTRASONIC["facing"] * approach   # approach speed -> belt x direction
    return [dict(id="US", label="ultrasonic", x_cm=x, speed_cm_s=speed, expected_cm=None, err_cm=None, diverging=False)]


def objects_view():
    head = html.Tr([html.Th(h) for h in ("Id", "Class", "x (cm)", "v (cm/s)", "Δ pred. (cm)")])
    rows = []
    ultrasonic = state.params.get("pos_source") == "ultrasonic"
    for o in sorted(belt_objects(), key=lambda o: str(o["id"])):
        sp = "—" if o["speed_cm_s"] is None else f"{o['speed_cm_s']:+.1f}"
        er = "—" if o["err_cm"] is None else f"{o['err_cm']:+.1f}"
        rows.append(html.Tr(className="bad" if o.get("diverging") else "", children=[
            html.Td(o["id"]), html.Td(o["label"]), html.Td(f"{o['x_cm']:.1f}"), html.Td(sp), html.Td(er)]))
    if not rows:
        rows = [html.Tr(html.Td("No objects detected on the belt", colSpan=5, className="empty"))]
    title = "Object seen by the ultrasonic sensor" if ultrasonic else "Objects seen by the camera"
    return card(title, html.Table(className="tbl", children=[html.Thead(head), html.Tbody(rows)]))


def events_view():
    rows = []
    for t, lvl, msg in list(state.events)[:14]:
        rows.append(html.Div(className=f"ev {lvl}", children=[
            html.Span(time.strftime("%H:%M:%S", time.localtime(t)), className="t"), html.Span(msg)]))
    return rows or [html.Div("No events yet", className="empty")]


def monitor_view():
    now = time.time()
    rows = [html.Tr([html.Td(tp), html.Td(pl[:90]), html.Td(f"{now - t:.1f} s")])
            for tp, (t, pl) in sorted(state.raw.items())]
    return html.Table(className="tbl", children=[
        html.Thead(html.Tr([html.Th("Topic"), html.Th("Last payload"), html.Th("Age")])), html.Tbody(rows)])


def badges_view():
    a = state.age("rpm")
    if not state.mqtt_connected:
        mq = badge("MQTT (camera)", "broker unreachable", "alarm")
    else:
        mq = badge("MQTT (camera)", f"{cfg.MQTT['host']}", "ok")
    if a is not None and a < cfg.STALE_AFTER_S:
        dev = badge("Conveyor", f"live · {a:.1f} s ago", "ok")
    elif state.device_status == "online":
        dev = badge("Conveyor", "online, no telemetry", "warn")
    else:
        dev = badge("Conveyor", "no data", "alarm")
    vs = state.vision_status
    cam = badge("Camera", vs, "ok" if vs in ("streaming", "simulated") else ("na" if vs == "off" else "warn"))
    return [mq, dev, cam]


# ════════════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════════════
def belt_figure():
    sandbox = not state.sync
    speed = state.get("model_belt_speed_cm_s") if sandbox else state.get("belt_speed_cm_s")
    travel = state.travel_model if sandbox else state.travel_real
    off = travel % 5.0

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=L, y0=1, y1=5, fillcolor="#37414f", line=dict(color=INK, width=2), layer="below")
    fig.add_shape(type="line", x0=0, x1=L, y0=5.7, y1=5.7, line=dict(color=PURPLE, width=1, dash="dot"))

    # rollers
    fig.add_trace(go.Scatter(x=[0, L], y=[3, 3], mode="markers", hoverinfo="skip",
                             marker=dict(size=92, color="#c9d1db", line=dict(color=INK, width=3))))
    # moving stripes = belt motion
    xs, ys = [], []
    k = 0
    while off + 5 * k < L:
        p = off + 5 * k
        if p > 0.5:
            xs += [p, p, None]
            ys += [1.2, 4.8, None]
        k += 1
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", line=dict(color="#556173", width=2)))

    # ultrasonic sensor position (it may sit past the end of the belt)
    ux = cfg.ULTRASONIC["sensor_x_cm"]
    fig.add_trace(go.Scatter(x=[ux], y=[3], mode="markers+text", text=["US"], textposition="bottom center",
                             hoverinfo="skip", textfont=dict(color=INK, size=11),
                             marker=dict(symbol="triangle-left" if cfg.ULTRASONIC["facing"] < 0 else "triangle-right",
                                         size=18, color=GREEN, line=dict(color=INK, width=1))))

    objs = belt_objects()
    # expected positions (predicted from encoder travel)
    ex = [o for o in objs if o["err_cm"] is not None and abs(o["err_cm"]) > 0.5]
    if ex:
        fig.add_trace(go.Scatter(x=[o["expected_cm"] for o in ex], y=[3] * len(ex), mode="markers", name="Expected (encoder)",
                                 hoverinfo="skip", marker=dict(symbol="square-open", size=46, color=AMBER, line=dict(width=3))))
    for bad, col in ((False, GREEN), (True, RED)):
        sel = [o for o in objs if bool(o.get("diverging")) == bad]
        if sel:
            txt = [f"#{o['id']} {o['label']}<br>" + ("" if o["speed_cm_s"] is None else f"{o['speed_cm_s']:+.1f} cm/s") for o in sel]
            fig.add_trace(go.Scatter(x=[o["x_cm"] for o in sel], y=[3] * len(sel), mode="markers+text", text=txt,
                                     textposition="top center", textfont=dict(color="#fff", size=12), hoverinfo="skip",
                                     marker=dict(symbol="square", size=34, color=col, line=dict(color="#fff", width=2))))

    # direction / speed
    if speed is None:
        head, col = "waiting for speed data", GREY
    elif abs(speed) < 0.3:
        head, col = "Stopped", GREY
    elif speed > 0:
        head, col = f"▶ ▶ ▶   Forward   {speed:.1f} cm/s", COBALT
    else:
        head, col = f"◀ ◀ ◀   Reverse   {abs(speed):.1f} cm/s", COBALT
    fig.add_annotation(x=L / 2, y=7.5, text=f"<b>{head}</b>", showarrow=False, font=dict(size=20, color=col))
    src = "ultrasonic sensor" if state.params.get("pos_source") == "ultrasonic" else f"camera (calibrated 0 – {L:g} cm)"
    fig.add_annotation(x=0, y=6.2, text=f"object position from: {src}", showarrow=False, xanchor="left",
                       font=dict(size=11, color=PURPLE))
    if sandbox:
        fig.add_annotation(x=0, y=7.5, text="Sandbox: showing the twin model", showarrow=False, xanchor="left",
                           font=dict(size=12, color=AMBER))

    fig.update_layout(
        showlegend=False, margin=dict(l=10, r=10, t=4, b=44), uirevision="belt",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        xaxis=dict(range=[min(-6, ux - 4), max(L + 6, ux + 4)], tickvals=list(range(0, int(L) + 1, 5)), ticksuffix=" cm", showgrid=False,
                   zeroline=False, ticks="outside", linecolor=INK, fixedrange=True),
        yaxis=dict(range=[0, 8.2], visible=False, fixedrange=True))
    return fig


def plot_figure(p):
    now = time.time()
    palette = [COBALT, GREY, AMBER, PURPLE, GREEN]
    fig = go.Figure()
    for i, vid in enumerate(p["series"]):
        v = VAR.get(vid)
        if v is None:
            continue
        pts = [(t - now, y) for t, y in state.series(vid) if now - t <= cfg.HISTORY_SECONDS]
        dash = {"model": "dash", "vision": "dot"}.get(v.get("source"), "solid")
        fig.add_trace(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="lines", name=v["label"],
                                 line=dict(color=v.get("color", palette[i % len(palette)]), width=2, dash=dash)))
    fig.update_layout(
        title=dict(text=p["title"], x=0.01, font=dict(size=14)), height=260, margin=dict(l=48, r=12, t=40, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)", font=dict(color=INK, size=12),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"), uirevision="plots",
        xaxis=dict(range=[-cfg.HISTORY_SECONDS, 0], title="seconds ago", gridcolor="#d5dde5"),
        yaxis=dict(gridcolor="#d5dde5"))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Layout
# ════════════════════════════════════════════════════════════════════════════
def widget(c):
    cid = {"type": "ctl", "id": c["id"]}
    val = controls.to_component(c, state.controls[c["id"]])
    k = c["kind"]
    if k == "slider":
        return dcc.Slider(id=cid, min=c["min"], max=c["max"], step=c.get("step", 1), value=val, marks=None,
                          tooltip={"placement": "bottom", "always_visible": True}, updatemode="mouseup")
    if k == "buttons":
        return dcc.RadioItems(id=cid, options=[{"label": l, "value": v} for l, v in c["options"]], value=val,
                              className="seg", inline=True)
    if k == "number":
        return dcc.Input(id=cid, type="number", value=val, step=c.get("step", 1), min=c.get("min"), max=c.get("max"),
                         debounce=True, className="num")
    if k == "switch":
        return dcc.Checklist(id=cid, options=[{"label": " on", "value": "on"}], value=val, className="sw")
    raise ValueError(f"unknown control kind {k}")


def controls_block():
    groups = {}
    for c in cfg.CONTROLS:
        groups.setdefault(c.get("group", "Controls"), []).append(c)
    tags = {"real": "real", "twin": "twin only", "both": "real + twin", "camera": "ESP32-CAM"}
    out = []
    for g, items in groups.items():
        rows = []
        for c in items:
            tag = tags.get(c.get("target", "real"), "")
            rows.append(html.Div(className="ctl", children=[
                html.Div(className="ctl-h", children=[html.Span(c["label"]), html.Span(tag, className=f"tag {c.get('target', 'real')}")]),
                widget(c)]))
        out.append(card(g, *rows))
    return out


def serve_layout():
    return html.Div(className="app", children=[
        html.Header(className="top", children=[
            html.Div(children=[html.H1("Conveyor digital twin"),
                               html.P(f"{L:g} cm belt · Arduino over serial · camera + bg subtraction", className="sub")]),
            html.Div(id="badges", className="badges"),
            html.Div(className="actions", children=[
                dcc.RadioItems(id="sync", value="live" if state.sync else "sandbox", className="seg", inline=True,
                               options=[{"label": "Live sync", "value": "live"}, {"label": "Sandbox", "value": "sandbox"}]),
                html.Button("E-stop", id="estop", className="estop"),
            ]),
        ]),
        html.Section(className="twin", children=[
            dcc.Graph(id="belt", config={"displayModeBar": False}, style={"height": "310px"}),
            html.Div(id="last-cmd", className="last-cmd"),
        ]),
        html.Main(className="cols", children=[
            html.Div(className="col", children=controls_block()),
            html.Div(className="col", children=[html.Div(id="panels", className="stack"), html.Div(id="diverge")]),
            html.Div(className="col", children=[
                card("Camera", html.Img(src="/video_feed", className="cam")),
                html.Div(id="objects"),
            ]),
        ]),
        html.Section(className="plots", children=[
            dcc.Graph(id={"type": "plot", "id": p["id"]}, config={"displayModeBar": False}) for p in cfg.PLOTS]),
        html.Section(className="logs", children=[
            card("Events", html.Div(id="events")),
            card("Comm monitor", html.Div(id="monitor")),
        ]),
        dcc.Interval(id="fast", interval=250),
        dcc.Interval(id="slow", interval=500),
        html.Div(id="ctl-sink", hidden=True),
        html.Div(id="sync-sink", hidden=True),
    ])


controls.init()            # defaults must exist before the layout is built
app.layout = serve_layout


# ════════════════════════════════════════════════════════════════════════════
# Callbacks
# ════════════════════════════════════════════════════════════════════════════
@app.callback(Output("ctl-sink", "children"), Input({"type": "ctl", "id": ALL}, "value"), prevent_initial_call=True)
def on_control(_):
    trig = ctx.triggered_id
    if not trig:
        return no_update
    c = controls.BY_ID.get(trig["id"])
    typed = controls.parse(c, ctx.triggered[0]["value"]) if c else None
    if typed is None:
        return no_update
    controls.apply(c, typed, bridge)
    return ""


@app.callback(Output({"type": "ctl", "id": cfg.ESTOP["control"]}, "value"), Input("estop", "n_clicks"), prevent_initial_call=True)
def on_estop(_):
    c = controls.BY_ID[cfg.ESTOP["control"]]
    state.log("E-STOP pressed", "alert")
    controls.apply(c, cfg.ESTOP["value"], bridge, force=True)
    return controls.to_component(c, cfg.ESTOP["value"])


@app.callback(Output("sync-sink", "children"), Input("sync", "value"))
def on_sync(v):
    live = v == "live"
    if live != state.sync:
        state.sync = live
        state.log("Live sync: commands go to the real conveyor" if live
                  else "Sandbox: changes only affect the digital twin", "warn")
    return ""


@app.callback(
    Output("badges", "children"), Output("belt", "figure"), Output("panels", "children"), Output("diverge", "children"),
    Output("objects", "children"), Output("events", "children"), Output("monitor", "children"), Output("last-cmd", "children"),
    Input("fast", "n_intervals"))
def refresh(_):
    return (badges_view(), belt_figure(), panels_view(), divergence_view(), objects_view(), events_view(),
            monitor_view(), "Last command: " + state.last_command)


@app.callback(Output({"type": "plot", "id": ALL}, "figure"), Input("slow", "n_intervals"))
def refresh_plots(_):
    return [plot_figure(p) for p in cfg.PLOTS]


# ════════════════════════════════════════════════════════════════════════════
def main():
    bridge.start()
    mqtt.start()
    twin_model.start()
    vision.start(mqtt)
    url = f"http://{cfg.WEB['host']}:{cfg.WEB['port']}"
    print(f"\n  Digital twin running at {url}\n")
    if cfg.WEB.get("open_browser"):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # threaded=True: /video_feed is an infinite streaming connection: without this, Flask's dev server
    # (single request at a time by default) gets stuck serving it and everything else - the periodic
    # belt/panels updates included - queues behind it forever.
    # debug=False: the reloader would start every thread (serial reader, MQTT client, vision) twice.
    app.run(host=cfg.WEB["host"], port=cfg.WEB["port"], debug=False, threaded=True)


if __name__ == "__main__":
    main()
