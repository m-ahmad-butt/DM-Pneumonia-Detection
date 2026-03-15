import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import math

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV Demo")
        self.root.state('zoomed')
        self.root.configure(bg="#1a1b26")

        self.original_image = None
        self.current_processed = None
        self.display_image = None
        self.patch_size = 40

        self.params = {
            "blur_type": "Gaussian",
            "k_size": 5,
            "sigma": 1.5,
            "clahe_clip": 2.0,
            "color_mode": "BGR (Original)",
            "norm_alpha": 0,
            "norm_beta": 255,
            "resize_mode": "Linear",
            "resize_factor": 1.0
        }

        self.setup_ui()

    def setup_ui(self):
        header = tk.Label(self.root, text="Computer Vision Theory & Lab", font=("Segoe UI", 24, "bold"), bg="#1a1b26", fg="#7aa2f7")
        header.pack(fill=tk.X, pady=10)

        main_container = tk.Frame(self.root, bg="#1a1b26")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        sidebar = tk.Frame(main_container, bg="#16161e", width=220, padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="MODULES", font=("Segoe UI", 12, "bold"), bg="#16161e", fg="#565f89").pack(pady=(0, 10))
        
        self.modules = ["📂 File Loader", "🎨 Color Spaces", "🌫️ Smoothing", "📊 Histogram & CLAHE", "⚖️ Normalization", "📐 Resizing", "🔳 Edges", "✨ Effects"]
        self.module_list = tk.Listbox(sidebar, bg="#1a1b26", fg="#c0caf5", selectbackground="#3d59a1", font=("Segoe UI", 11), borderwidth=0, highlightthickness=0, exportselection=tk.FALSE)
        for m in self.modules: self.module_list.insert(tk.END, m)
        self.module_list.pack(fill=tk.BOTH, expand=True)
        self.module_list.bind("<<ListboxSelect>>", self.on_module_change)
        self.module_list.select_set(0)

        action_f = tk.Frame(sidebar, bg="#16161e", pady=10)
        action_f.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Button(action_f, text="COMMIT Session", command=self.save_session, bg="#9ece6a", fg="#1a1b26", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=8).pack(fill=tk.X, pady=5)
        tk.Button(action_f, text="VIEW CODE", command=self.show_code_window, bg="#3d59a1", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=8).pack(fill=tk.X, pady=5)
        tk.Button(action_f, text="RESET ALL", command=self.reset_image, bg="#f7768e", fg="#1a1b26", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=8).pack(fill=tk.X, pady=5)

        center_frame = tk.Frame(main_container, bg="#1a1b26")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)

        self.param_panel = tk.Frame(center_frame, bg="#16161e", height=240, padx=20, pady=20, relief=tk.RIDGE, borderwidth=1)
        self.param_panel.pack(fill=tk.X, pady=(0, 10))
        self.param_panel.pack_propagate(False)

        canvas_bg = tk.Frame(center_frame, bg="#24283b")
        canvas_bg.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Label(canvas_bg, bg="#24283b", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        code_frame = tk.Frame(center_frame, bg="#16161e", height=100, padx=15, pady=10, relief=tk.SUNKEN, borderwidth=1)
        code_frame.pack(fill=tk.X, pady=(10, 0))
        code_frame.pack_propagate(False)
        tk.Label(code_frame, text="PYTHON CODE SNIPPET", font=("Segoe UI", 9, "bold"), bg="#16161e", fg="#9ece6a").pack(anchor=tk.W)
        self.code_box = tk.Text(code_frame, height=2, bg="#1a1b26", fg="#73daca", font=("Consolas", 10), relief=tk.FLAT, padx=10, pady=5)
        self.code_box.pack(fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(main_container, bg="#16161e", width=420, padx=15, pady=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="LIVE HISTOGRAM", font=("Segoe UI", 10, "bold"), bg="#16161e", fg="#7aa2f7").pack()
        self.hist_canvas = tk.Canvas(right_panel, width=380, height=180, bg="#1a1b26", highlightthickness=0)
        self.hist_canvas.pack(pady=5)

        tk.Label(right_panel, text="PIXEL MAGNIFIER", font=("Segoe UI", 10, "bold"), bg="#16161e", fg="#bb9af7").pack(pady=(10, 0))
        self.zoom_canvas = tk.Label(right_panel, bg="#1a1b26", width=250, height=250, relief=tk.SOLID)
        self.zoom_canvas.pack(pady=5)

        tk.Label(right_panel, text="THEORY & MATH LAB", font=("Segoe UI", 10, "bold"), bg="#16161e", fg="#e0af68").pack(pady=(10, 0))
        self.theory_box = tk.Text(right_panel, height=8, width=50, bg="#1f2335", fg="#cfc9c2", font=("Consolas", 9), relief=tk.FLAT, padx=10, pady=10)
        self.theory_box.pack(pady=5)

        tk.Label(right_panel, text="3x3 INTENSITY MATRIX", font=("Segoe UI", 10, "bold"), bg="#16161e", fg="#9ece6a").pack(pady=(10, 0))
        self.matrix_box = tk.Text(right_panel, height=5, width=50, bg="#1f2335", fg="#73daca", font=("Consolas", 10), relief=tk.FLAT, padx=10, pady=5)
        self.matrix_box.pack()

        self.on_module_change(None)

    def on_module_change(self, event):
        idx = self.module_list.curselection()
        if not idx: return 
        mod = self.modules[idx[0]]
        
        for w in self.param_panel.winfo_children(): w.destroy()

        if "File" in mod:
            tk.Button(self.param_panel, text="OPEN NEW IMAGE FILE", command=self.load_image, bg="#7aa2f7", fg="#1a1b26", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, padx=30, pady=15).pack(expand=True)
            self.update_theory("Data Loading", "Images are loaded as BGR by default in OpenCV.\n\nTip: Use 'COMMIT' on the left to lock in your current filters permanently.")

        elif "Color" in mod:
            tk.Label(self.param_panel, text="Color Models Lab", font=("Segoe UI", 12, "bold"), bg="#16161e", fg="#7aa2f7").pack(pady=5)
            self.color_var = tk.StringVar(value=self.params["color_mode"] if self.params["color_mode"] in ["RGB (Full Color)", "Grayscale"] else "RGB (Full Color)")
            cb = ttk.Combobox(self.param_panel, textvariable=self.color_var, values=["RGB (Full Color)", "Grayscale"], state="readonly")
            cb.pack(fill=tk.X)
            cb.bind("<<ComboboxSelected>>", self.refresh)
            self.update_theory("Color Conversions", "Grayscale: Y = 0.299R + 0.587G + 0.114B\n\nRGB: Uses 3 channels representing Red, Green, and Blue light additive model.")

        elif "Smoothing" in mod:
            self.blur_type = tk.StringVar(value=self.params["blur_type"])
            f = tk.Frame(self.param_panel, bg="#16161e")
            f.pack()
            tk.Radiobutton(f, text="Gaussian", variable=self.blur_type, value="Gaussian", command=self.refresh, bg="#16161e", fg="#c0caf5", selectcolor="#1a1b26").pack(side=tk.LEFT, padx=10)
            tk.Radiobutton(f, text="Median", variable=self.blur_type, value="Median", command=self.refresh, bg="#16161e", fg="#c0caf5", selectcolor="#1a1b26").pack(side=tk.LEFT)
            
            self.ks_scale = tk.Scale(self.param_panel, from_=1, to=31, resolution=2, label="Kernel Size (Odd)", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.ks_scale.set(self.params["k_size"])
            self.ks_scale.pack(fill=tk.X)

            self.sig_scale = tk.Scale(self.param_panel, from_=0.1, to=10.0, resolution=0.1, label="Sigma (σ)", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.sig_scale.set(self.params["sigma"])
            self.sig_scale.pack(fill=tk.X)

        elif "Histogram" in mod:
            tk.Label(self.param_panel, text="Contrast Enhancement Laboratory", bg="#16161e", fg="#7aa2f7").pack()
            self.eq_type = tk.StringVar(value="CLAHE")
            opt = tk.OptionMenu(self.param_panel, self.eq_type, "Standard", "CLAHE", "None", command=lambda x: self.refresh())
            opt.config(bg="#1a1b26", fg="#c0caf5", activebackground="#3d59a1", highlightthickness=0)
            opt.pack(pady=10)
            
            self.clip_scale = tk.Scale(self.param_panel, from_=1.0, to=10.0, resolution=0.1, label="CLAHE Clip Limit", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.clip_scale.set(self.params["clahe_clip"])
            self.clip_scale.pack(fill=tk.X)
            self.update_theory("CLAHE", "Adaptive Histogram Equalization divides image into blocks.\n\nClip Limit: Avoids amplifying noise in uniform areas by clipping the peaks of the local histogram.")

        elif "Normalization" in mod:
            tk.Label(self.param_panel, text="Min-Max Scaling [a, b]", bg="#16161e", fg="#7aa2f7").pack()
            self.n_a = tk.Scale(self.param_panel, from_=0, to=255, label="Alpha (Target Min)", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.n_a.set(self.params["norm_alpha"])
            self.n_a.pack(fill=tk.X)
            self.n_b = tk.Scale(self.param_panel, from_=0, to=255, label="Beta (Target Max)", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.n_b.set(self.params["norm_beta"])
            self.n_b.pack(fill=tk.X)

        elif "Resizing" in mod:
            tk.Label(self.param_panel, text="Interpolation Strategy", bg="#16161e", fg="#7aa2f7").pack()
            self.interp_var = tk.StringVar(value=self.params["resize_mode"])
            cb = ttk.Combobox(self.param_panel, textvariable=self.interp_var, values=["Linear", "Cubic", "Nearest", "Area"], state="readonly")
            cb.pack(fill=tk.X, pady=5)
            cb.bind("<<ComboboxSelected>>", self.refresh)
            
            self.f_scale = tk.Scale(self.param_panel, from_=0.1, to=2.0, resolution=0.1, label="Scaling Factor (Preview)", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.f_scale.set(self.params["resize_factor"])
            self.f_scale.pack(fill=tk.X)
            
            self.update_theory("Interpolation Theory", "Linear: Bi-linear weighted average.\nCubic: Bi-cubic 16-pixel neighborhood.\nNearest: Pick closest pixel (Fastest).\nArea: Resampling using pixel area relation.")

        elif "Edges" in mod:
            self.e_l = tk.Scale(self.param_panel, from_=0, to=255, label="Low Threshold", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.e_l.set(100); self.e_l.pack(fill=tk.X)
            self.e_h = tk.Scale(self.param_panel, from_=0, to=255, label="High Threshold", orient=tk.HORIZONTAL, bg="#16161e", fg="#c0caf5", command=lambda x: self.refresh())
            self.e_h.set(200); self.e_h.pack(fill=tk.X)

        elif "Effects" in mod:
            tk.Label(self.param_panel, text="Custom Kernel Filtering (Sharpen)", bg="#16161e", fg="#7aa2f7").pack()
            grid_f = tk.Frame(self.param_panel, bg="#16161e")
            grid_f.pack(pady=5)
            self.k_entries = []
            default_k = [[0,-1,0], [-1,5,-1], [0,-1,0]]
            for r in range(3):
                row_es = []
                for c in range(3):
                    e = tk.Entry(grid_f, width=4, justify='center', bg="#1a1b26", fg="#73daca", insertbackground="white")
                    e.insert(0, str(default_k[r][c]))
                    e.grid(row=r, column=c, padx=2, pady=2)
                    row_es.append(e)
                self.k_entries.append(row_es)
            
            tk.Button(self.param_panel, text="APPLY KERNEL", command=self.refresh, bg="#3d59a1", fg="white", relief=tk.FLAT).pack(pady=5)
            
            self.glow_v = tk.BooleanVar(value=False); tk.Checkbutton(self.param_panel, text="Glow Effect (Bloom)", variable=self.glow_v, command=self.refresh, bg="#16161e", fg="#c0caf5", selectcolor="#1a1b26").pack(anchor=tk.W)

        self.refresh()

    def load_image(self):
        p = filedialog.askopenfilename()
        if p:
            self.raw_file_path = p
            self.original_image = cv2.imread(p)
            self.refresh()

    def refresh(self, *args):
        if self.original_image is None: return
        img = self.original_image.copy()
        mod = self.module_list.get(self.module_list.curselection()[0]) if self.module_list.curselection() else "📂 File Loader"

        if "Color" in mod:
            self.params["color_mode"] = self.color_var.get()
            m = self.params["color_mode"]
            if m == "Grayscale": 
                if len(img.shape) == 3: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                if len(img.shape) == 3: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif "Smoothing" in mod:
            self.params["blur_type"] = self.blur_type.get()
            self.params["k_size"] = int(self.ks_scale.get())
            self.params["sigma"] = float(self.sig_scale.get())
            k = self.params["k_size"]
            if self.params["blur_type"] == "Gaussian":
                img = cv2.GaussianBlur(img, (k, k), self.params["sigma"])
                self.calc_gaus_kernel(k, self.params["sigma"])
            else:
                img = cv2.medianBlur(img, k)
        elif "Histogram" in mod:
            self.params["clahe_clip"] = float(self.clip_scale.get())
            mode = self.eq_type.get()
            if mode == "Standard": img = self.bgr_equalize(img)
            elif mode == "CLAHE": img = self.bgr_clahe(img, self.params["clahe_clip"])
        elif "Normalization" in mod:
            self.params["norm_alpha"] = int(self.n_a.get())
            self.params["norm_beta"] = int(self.n_b.get())
            img = cv2.normalize(img, None, self.params["norm_alpha"], self.params["norm_beta"], cv2.NORM_MINMAX)
        elif "Resizing" in mod:
            self.params["resize_mode"] = self.interp_var.get()
            self.params["resize_factor"] = float(self.f_scale.get())
            im_m = {"Linear": cv2.INTER_LINEAR, "Cubic": cv2.INTER_CUBIC, "Nearest": cv2.INTER_NEAREST, "Area": cv2.INTER_AREA}
            s = self.params["resize_factor"]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=im_m[self.params["resize_mode"]])
        elif "Edges" in mod:
            img = cv2.Canny(img, self.e_l.get(), self.e_h.get())
        elif "Effects" in mod:
            try:
                kernel = np.array([[float(e.get()) for e in row] for row in self.k_entries], dtype=np.float32)
                img = cv2.filter2D(img, -1, kernel)
            except: pass 
            
            if self.glow_v.get():
                b = cv2.GaussianBlur(img, (25, 25), 0)
                img = cv2.addWeighted(img, 1.0, b, 0.5, 0)

        self.current_processed = img
        self.update_displays()
        self.generate_code_snippet(mod)

    def generate_code_snippet(self, mod):
        self.code_box.delete("1.0", tk.END)
        code = "# Standard Op\n"
        
        if "Color" in mod:
            m = self.params["color_mode"]
            if m == "Grayscale": code = "gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"
            else: code = "rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)"
        
        elif "Smoothing" in mod:
            k = self.params["k_size"]; sig = self.params["sigma"]
            if self.params["blur_type"] == "Gaussian":
                code = f"blurred = cv2.GaussianBlur(img, ({k}, {k}), {sig})"
            else:
                code = f"blurred = cv2.medianBlur(img, {k})"
                
        elif "Histogram" in mod:
            mode = self.eq_type.get()
            clip = self.params["clahe_clip"]
            if mode == "Standard": code = "yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)\nyuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])\nimg = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)"
            else: code = f"clahe = cv2.createCLAHE(clipLimit={clip}, tileGridSize=(8,8))\nimg_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)\nimg_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])\nimg = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)"
            
        elif "Normalization" in mod:
            a = self.params["norm_alpha"]; b = self.params["norm_beta"]
            code = f"norm = cv2.normalize(img, None, {a}, {b}, cv2.NORM_MINMAX)"
            
        elif "Resizing" in mod:
            s = self.params["resize_factor"]
            mode = self.params["resize_mode"]
            im_m = {"Linear": "cv2.INTER_LINEAR", "Cubic": "cv2.INTER_CUBIC", "Nearest": "cv2.INTER_NEAREST", "Area": "cv2.INTER_AREA"}
            code = f"resized = cv2.resize(img, None, fx={s}, fy={s}, interpolation={im_m[mode]})"
            
        elif "Edges" in mod:
            l = self.e_l.get(); h = self.e_h.get()
            code = f"edges = cv2.Canny(img, {l}, {h})"
            
        elif "Effects" in mod:
            kernel = [[float(e.get()) for e in row] for row in self.k_entries]
            code = f"kernel = np.array({kernel}, dtype=np.float32)\nimg = cv2.filter2D(img, -1, kernel)"

        self.code_box.insert(tk.END, code)

    def show_code_window(self):
        snippet = self.code_box.get("1.0", tk.END)
        if not snippet.strip() or "Standard Op" in snippet:
            tk.messagebox.showwarning("No Selection", "Please select a processing module and adjust parameters first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Python Code Export")
        win.state('zoomed') 
        win.configure(bg="#1a1b26")
        
        tk.Label(win, text="Ready-to-use OpenCV Code:", font=("Segoe UI", 12, "bold"), bg="#1a1b26", fg="#7aa2f7", pady=10).pack()
        txt = tk.Text(win, bg="#16161e", fg="#73daca", font=("Consolas", 11), padx=15, pady=15, relief=tk.FLAT)
        txt.insert(tk.END, snippet)
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

    def save_session(self):
        if self.current_processed is not None:
            self.original_image = self.current_processed.copy()
            tk.messagebox.showinfo("Session Saved", "Current changes committed. All subsequent filters will apply to this new base image.")
            self.refresh()

    def reset_image(self):
        if hasattr(self, 'raw_file_path'):
            self.original_image = cv2.imread(self.raw_file_path)
            self.refresh()
            tk.messagebox.showinfo("Reset", "Reverted to original file state.")

    def bgr_equalize(self, img):
        if len(img.shape) == 2: return cv2.equalizeHist(img)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def bgr_clahe(self, img, clip):
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        if len(img.shape) == 2: return c.apply(img)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = c.apply(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def calc_gaus_kernel(self, k, sig):
        raw = cv2.getGaussianKernel(k, sig)
        m2d = np.outer(raw, raw)
        self.params["center_w_val"] = m2d[k//2, k//2]
        self.update_theory("Gaussian Kernel Matrix", f"Center weight (normalized): {m2d[k//2, k//2]:.4f}\n\nMatrix 3x3 Snippet:\n{m2d[k//2-1:k//2+2, k//2-1:k//2+2]}")

    def update_displays(self):
        disp = self.current_processed.copy()
        if len(disp.shape) == 2: disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        else: disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        
        h, w = disp.shape[:2]; max_s = 600
        if h > max_s or w > max_s:
            r = max_s / max(h, w); disp = cv2.resize(disp, (int(w*r), int(h*r)))
        
        self.last_disp = disp
        pil = Image.fromarray(disp)
        self.display_image = ImageTk.PhotoImage(image=pil)
        self.canvas.config(image=self.display_image)
        self.update_histogram(self.current_processed)

    def update_histogram(self, img):
        self.hist_canvas.delete("all")
        if len(img.shape) == 2:
            h = cv2.calcHist([img], [0], None, [256], [0,256])
            self.plot_hist(h, "#c0caf5")
        else:
            for i, col in enumerate(["#7aa2f7", "#9ece6a", "#f7768e"]):
                h = cv2.calcHist([img], [i], None, [256], [0,256])
                self.plot_hist(h, col)

    def plot_hist(self, h, color):
        cv2.normalize(h, h, 0, 160, cv2.NORM_MINMAX)
        points = []
        for i in range(256):
            points.extend([(i/255)*380, 180 - h[i][0]])
        self.hist_canvas.create_line(points, fill=color, width=2)

    def update_theory(self, title, txt):
        self.theory_box.delete("1.0", tk.END)
        self.theory_box.insert(tk.END, f"--- {title} ---\n\n{txt}")

    def on_mouse_move(self, event):
        if not hasattr(self, 'last_disp'): return
        h, w = self.last_disp.shape[:2]; cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = event.x - (cw-w)//2, event.y - (ch-h)//2
        if 0 <= x < w and 0 <= y < h:
            self.update_zoom(x, y)
            self.matrix_box.delete("1.0", tk.END)
            r1, r2 = max(0, y-1), min(h, y+2); c1, c2 = max(0, x-1), min(w, x+2)
            block = self.last_disp[r1:r2, c1:c2]
            for row in block:
                self.matrix_box.insert(tk.END, f"{[[p[0], p[1], p[2]] for p in row]}\n")
            self.update_live_theory(x, y)

    def update_zoom(self, x, y):
        p = 25; h, w = self.last_disp.shape[:2]
        x1, x2 = max(0, x-p), min(w, x+p); y1, y2 = max(0, y-p), min(h, y+p)
        patch = self.last_disp[y1:y2, x1:x2]
        
        im_m = {"Linear": cv2.INTER_LINEAR, "Cubic": cv2.INTER_CUBIC, "Nearest": cv2.INTER_NEAREST, "Area": cv2.INTER_AREA}
        mode = im_m.get(self.params.get("resize_mode", "Linear"), cv2.INTER_LINEAR)
        
        zoom = cv2.resize(patch, (250, 250), interpolation=mode)
        cv2.rectangle(zoom, (120, 120), (130, 130), (0,255,0), 1)
        pil = Image.fromarray(zoom); self.z_photo = ImageTk.PhotoImage(image=pil)
        self.zoom_canvas.config(image=self.z_photo)

    def update_live_theory(self, x, y):
        mod_sel = self.module_list.curselection()
        mod_text = self.modules[mod_sel[0]] if mod_sel else ""
        
        px = self.last_disp[y, x]
        self.update_theory("Live Pixel Calculus", f"Pixel Coord: ({x}, {y})\nRGB values: {px[0]}, {px[1]}, {px[2]}\n\n")
        
        if "Smoothing" in mod_text:
            k = self.params["k_size"]
            if self.params["blur_type"] == "Gaussian":
                self.theory_box.insert(tk.END, f"Gaussian Convolution:\nEvaluating {k}x{k} grid weights.\nCentral Weight: {self.params.get('center_w_val', 0.0):.4f}")
            else:
                self.theory_box.insert(tk.END, f"Median Selection:\nSorting {k*k} neighborhood values.\nNew value: {np.median(self.last_disp[y-1:y+2, x-1:x+2, 0])}")
        elif "Edges" in mod_text:
            self.theory_box.insert(tk.END, "Detecting Gradient ∇f:\nMagnitude = sqrt(Gx² + Gy²)\nHigh thresholds keep strong edges.")
        elif "Effects" in mod_text:
            self.theory_box.insert(tk.END, "Sharpening weights:\n-1 -1 -1\n-1  9 -1\n-1 -1 -1")

if __name__ == "__main__":
    root = tk.Tk(); app = ImageProcessingApp(root); root.mainloop()
