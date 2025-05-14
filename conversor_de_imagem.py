import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps
from pathlib import Path
import os
import replicate
import requests
from dotenv import load_dotenv
import mediapipe as mp

# Carrega variáveis do .env
load_dotenv()

# Configurar tokens
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Inicializar MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class LivroColorirApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conversor - Livro de Colorir Fofo + IA")
        self.root.geometry("780x580")
        self.root.resizable(False, False)

        self.input_path = ""
        self.output_path = ""
        self.preview_image = None
        self.batch_images = []
        self.original_img = None
        self.processed_img = None

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        titulo = tk.Label(frame, text="Transformar Imagem em Desenho Fofo\n(Processamento Local Avançado)",
                          font=("Arial", 14, "bold"), justify="center")
        titulo.pack(pady=(0, 20))

        # Frame para imagens de entrada/saída
        preview_frame = tk.Frame(frame)
        preview_frame.pack(pady=10, fill=tk.X)
        
        # Labels para previews
        self.input_preview = tk.Label(preview_frame, text="Nenhuma imagem\nselecionada", 
                                     width=30, height=10, relief="ridge")
        self.input_preview.grid(row=0, column=0, padx=10)
        
        self.output_preview = tk.Label(preview_frame, text="Resultado\naparecerá aqui", 
                                      width=30, height=10, relief="ridge")
        self.output_preview.grid(row=0, column=1, padx=10)

        btn_imagem = tk.Button(frame, text="Selecionar Imagem",
                               font=("Arial", 11), width=30, command=self.selecionar_imagem)
        btn_imagem.pack(pady=5)

        btn_salvar = tk.Button(frame, text="Selecionar Local de Salvamento",
                               font=("Arial", 11), width=30, command=self.selecionar_salvar)
        btn_salvar.pack(pady=5)

        # Frame para opções de estilo
        options_frame = tk.Frame(frame)
        options_frame.pack(pady=10, fill=tk.X)

        # Estilo
        estilo_frame = tk.Frame(options_frame)
        estilo_frame.grid(row=0, column=0, pady=5, sticky='w')
        tk.Label(estilo_frame, text="Estilo de Processamento:", font=("Arial", 11)).pack(side=tk.LEFT)
        self.estilo_var = tk.StringVar(value="Advanced LineArt")
        estilo_combo = ttk.Combobox(estilo_frame,
                                    textvariable=self.estilo_var,
                                    values=[
                                        "Advanced LineArt",
                                        "Cartoon Contours",
                                        "Clean Contours",
                                        "Sketch Style",
                                        "Face Focus",
                                        "IA - Replicate LineArt",
                                        "IA - ControlNet HuggingFace"
                                    ],
                                    state="readonly", width=25)
        estilo_combo.pack(side=tk.LEFT, padx=10)
        estilo_combo.bind("<<ComboboxSelected>>", self.atualizar_preview_estilo)

        # Espessura
        espessura_frame = tk.Frame(options_frame)
        espessura_frame.grid(row=1, column=0, pady=5, sticky='w')
        tk.Label(espessura_frame, text="Espessura do Traço:", font=("Arial", 11)).pack(side=tk.LEFT)
        self.espessura_var = tk.StringVar(value="Médio")
        combo = ttk.Combobox(espessura_frame, textvariable=self.espessura_var,
                             values=["Fino", "Médio", "Grosso"], state="readonly", width=10)
        combo.pack(side=tk.LEFT, padx=10)
        combo.bind("<<ComboboxSelected>>", self.atualizar_preview_espessura)

        # Detalhe
        detalhe_frame = tk.Frame(options_frame)
        detalhe_frame.grid(row=2, column=0, pady=5, sticky='w')
        tk.Label(detalhe_frame, text="Nível de Detalhe:", font=("Arial", 11)).pack(side=tk.LEFT)
        self.detalhe_var = tk.StringVar(value="Médio")
        detalhe_combo = ttk.Combobox(detalhe_frame, textvariable=self.detalhe_var,
                                    values=["Baixo", "Médio", "Alto"], state="readonly", width=10)
        detalhe_combo.pack(side=tk.LEFT, padx=10)
        detalhe_combo.bind("<<ComboboxSelected>>", self.atualizar_preview_detalhe)

        # Suavidade
        suavidade_frame = tk.Frame(options_frame)
        suavidade_frame.grid(row=3, column=0, pady=5, sticky='w')
        tk.Label(suavidade_frame, text="Suavidade:", font=("Arial", 11)).pack(side=tk.LEFT)
        self.suavidade_var = tk.IntVar(value=50)
        suavidade_scale = ttk.Scale(suavidade_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.suavidade_var, length=200)
        suavidade_scale.pack(side=tk.LEFT, padx=10)
        suavidade_scale.bind("<ButtonRelease-1>", self.atualizar_preview_suavidade)

        # Opções de exportação
        export_frame = tk.Frame(frame)
        export_frame.pack(pady=10, fill=tk.X)
        
        self.exportar_pdf = tk.BooleanVar(value=True)
        chk_pdf = tk.Checkbutton(export_frame, text="Exportar também como PDF", variable=self.exportar_pdf)
        chk_pdf.pack(side=tk.LEFT, pady=5)
        
        self.inverter_cores = tk.BooleanVar(value=False)
        chk_inverter = tk.Checkbutton(export_frame, text="Inverter cores (fundo preto)", 
                                     variable=self.inverter_cores, command=self.atualizar_preview_cores)
        chk_inverter.pack(side=tk.LEFT, padx=20, pady=5)

        # Botões de ação
        btn_visualizar = tk.Button(frame, text="Visualizar em Tamanho Real", font=("Arial", 11),
                                   width=30, command=self.visualizar)
        btn_visualizar.pack(pady=5)

        btn_converter = tk.Button(frame, text="Converter e Salvar", font=("Arial", 11, "bold"),
                                  bg="#4CAF50", fg="white", width=30, command=self.converter_e_salvar)
        btn_converter.pack(pady=15)

        # Barra de status
        self.status_var = tk.StringVar(value="Pronto para iniciar")
        status_label = tk.Label(frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def atualizar_preview_estilo(self, event=None):
        self.atualizar_preview()
    
    def atualizar_preview_espessura(self, event=None):
        self.atualizar_preview()
        
    def atualizar_preview_detalhe(self, event=None):
        self.atualizar_preview()
        
    def atualizar_preview_suavidade(self, event=None):
        self.atualizar_preview()
        
    def atualizar_preview_cores(self):
        self.atualizar_preview()
    
    def atualizar_preview(self):
        if not hasattr(self, 'original_img') or self.original_img is None:
            return
            
        try:
            # Processar imagem em tamanho reduzido para preview
            preview_img = cv2.resize(self.original_img, (300, int(self.original_img.shape[0] * 300 / self.original_img.shape[1])))
            result = self.processar_imagem(preview_img, preview=True)
            
            # Aplicar inversão se necessário
            if self.inverter_cores.get():
                result = cv2.bitwise_not(result)
                
            # Atualizar preview
            result_pil = Image.fromarray(result)
            result_tk = ImageTk.PhotoImage(result_pil)
            self.output_preview.configure(image=result_tk)
            self.output_preview.image = result_tk
            
            # Salvar resultado para uso posterior
            self.processed_img = result
        except Exception as e:
            self.status_var.set(f"Erro no preview: {str(e)}")

    def selecionar_imagem(self):
        self.input_path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if self.input_path:
            # Carregar e exibir preview da imagem original
            self.original_img = cv2.imread(self.input_path)
            
            if self.original_img is not None:
                # Redimensionar para preview
                img_preview = cv2.resize(self.original_img, (300, int(self.original_img.shape[0] * 300 / self.original_img.shape[1])))
                img_preview_rgb = cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB)
                
                # Converter para formato tkinter
                img_pil = Image.fromarray(img_preview_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Atualizar label
                self.input_preview.configure(image=img_tk)
                self.input_preview.image = img_tk
                
                # Atualizar preview de saída
                self.atualizar_preview()
                
                # Sugerir nome de arquivo de saída
                nome_arquivo = os.path.splitext(os.path.basename(self.input_path))[0]
                self.output_path = os.path.join(os.path.dirname(self.input_path), f"{nome_arquivo}_colorir.png")
                
                self.status_var.set(f"Imagem carregada: {os.path.basename(self.input_path)}")

    def selecionar_salvar(self):
        if not self.input_path:
            messagebox.showwarning("Aviso", "Selecione uma imagem primeiro.")
            return

        nome_padrao = os.path.splitext(os.path.basename(self.input_path))[0] + "_colorir.png"
        self.output_path = filedialog.asksaveasfilename(
            title="Salvar como...",
            defaultextension=".png",
            initialfile=nome_padrao,
            filetypes=[("Imagem PNG", "*.png")]
        )
        
        if self.output_path:
            self.status_var.set(f"Local de salvamento definido: {os.path.basename(self.output_path)}")

    def visualizar(self):
        if not self.input_path:
            messagebox.showwarning("Aviso", "Selecione uma imagem primeiro.")
            return

        if self.processed_img is None:
            self.atualizar_preview()
            
        if self.processed_img is not None:
            # Criar uma nova janela para visualização em tamanho real
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Visualização em Tamanho Real")
            
            # Criar um Canvas com scrollbars para imagens grandes
            frame = tk.Frame(preview_window)
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Scrollbars
            h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
            v_scrollbar = tk.Scrollbar(frame)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Canvas
            canvas = tk.Canvas(frame, xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Configurar scrollbars
            h_scrollbar.config(command=canvas.xview)
            v_scrollbar.config(command=canvas.yview)
            
            # Converter imagem para formato tkinter
            img_pil = Image.fromarray(self.processed_img)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Adicionar imagem ao canvas
            canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
            canvas.image = img_tk  # Manter referência!
            
            # Configurar região de rolagem
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
            
            # Ajustar tamanho da janela
            img_width, img_height = img_pil.size
            screen_width = self.root.winfo_screenwidth() * 0.8
            screen_height = self.root.winfo_screenheight() * 0.8
            window_width = min(img_width + 30, screen_width)
            window_height = min(img_height + 30, screen_height)
            preview_window.geometry(f"{int(window_width)}x{int(window_height)}")

    def converter_e_salvar(self):
        if not self.input_path:
            messagebox.showerror("Erro", "Você precisa selecionar uma imagem.")
            return
            
        if not self.output_path:
            messagebox.showerror("Erro", "Você precisa selecionar o local de salvamento.")
            return

        try:
            self.status_var.set("Processando...")
            self.root.update()
            
            estilo = self.estilo_var.get()
            if "IA" in estilo:
                if "Replicate" in estilo:
                    imagem = self.processar_ia_replicate(self.input_path)
                elif "ControlNet" in estilo:
                    imagem = self.processar_ia_huggingface(self.input_path)
            else:
                imagem = self.processar_imagem(self.original_img)
                
            # Aplicar inversão se necessário
            if self.inverter_cores.get():
                imagem = cv2.bitwise_not(imagem)

            # Salvar imagem
            cv2.imwrite(self.output_path, imagem)

            # Exportar como PDF se solicitado
            if self.exportar_pdf.get():
                img_pil = Image.fromarray(imagem).convert("RGB")
                pdf_path = self.output_path.replace(".png", ".pdf")
                img_pil.save(pdf_path, "PDF")
                self.status_var.set(f"Imagem salva como PNG e PDF com sucesso!")
            else:
                self.status_var.set(f"Imagem salva como PNG com sucesso!")
                
            messagebox.showinfo("Sucesso", "Imagem processada e salva com sucesso!")
        except Exception as e:
            self.status_var.set(f"Erro: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao salvar: {str(e)}")

    def processar_imagem(self, img, preview=False):
        """Processa a imagem principal usando o estilo selecionado"""
        estilo = self.estilo_var.get()
        
        # Criar cópia para não modificar a original
        img_copy = img.copy()
        
        # Aplicar o estilo selecionado
        if estilo == "Advanced LineArt":
            resultado = self.estilo_lineart_avancado(img_copy)
        elif estilo == "Cartoon Contours":
            resultado = self.estilo_cartoon_contours(img_copy)
        elif estilo == "Clean Contours":
            resultado = self.estilo_contornos_limpos(img_copy)
        elif estilo == "Sketch Style":
            resultado = self.estilo_sketch(img_copy)
        elif estilo == "Face Focus":
            resultado = self.estilo_face_focus(img_copy)
        else:
            # Estilo padrão
            resultado = self.estilo_lineart_avancado(img_copy)
            
        return resultado

    def estilo_lineart_avancado(self, img):
        """Implementação avançada de extração de linhas tipo IA"""
        # Passo 1: Pré-processamento para reduzir ruído
        img_blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Passo 2: Detectar bordas com vários métodos e combiná-los
        # Converter para tons de cinza para processamento
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        
        # Ajustar parâmetros com base no nível de detalhe selecionado
        detalhe = self.detalhe_var.get()
        if detalhe == "Baixo":
            det_factor = 0.7
        elif detalhe == "Alto":
            det_factor = 1.3
        else:  # Médio
            det_factor = 1.0
            
        # Ajustar suavidade
        suavidade = self.suavidade_var.get() / 100.0  # Normalizar para 0-1
        
        # Canny com diferentes parâmetros
        canny_low1 = int(30 * det_factor)
        canny_high1 = int(150 * det_factor)
        edges_canny = cv2.Canny(gray, canny_low1, canny_high1)
        
        # Laplaciano para detalhe adicional
        laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        _, edges_laplacian = cv2.threshold(laplacian, 20 * det_factor, 255, cv2.THRESH_BINARY_INV)
        
        # Sobel para direções X e Y (destaca bordas em diferentes orientações)
        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
        sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
        _, edges_sobel = cv2.threshold(sobel_combined, 25 * det_factor, 255, cv2.THRESH_BINARY_INV)
        
        # Combinar as bordas de diferentes métodos com pesos
        edges_combined = cv2.addWeighted(edges_canny, 0.4, edges_laplacian, 0.3, 0)
        edges_combined = cv2.addWeighted(edges_combined, 0.8, edges_sobel, 0.2, 0)
        
        # Passo 3: Encontrar e desenhar contornos
        contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos muito pequenos (ruído)
        min_contour_area = 10 * det_factor
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        # Definir espessura do traço
        espessura = self.espessura_var.get()
        if espessura == "Fino":
            thickness = 1
        elif espessura == "Grosso":
            thickness = 3
        else:  # Médio
            thickness = 2
        
        # Desenhar contornos em uma imagem em branco
        contour_img = np.ones_like(gray) * 255
        cv2.drawContours(contour_img, filtered_contours, -1, (0), thickness)
        
        # Aplicar suavização aos contornos se necessário
        if suavidade > 0.1:
            blur_amount = int(5 * suavidade)
            if blur_amount % 2 == 0:  # garantir que seja ímpar
                blur_amount += 1
            contour_img = cv2.GaussianBlur(contour_img, (blur_amount, blur_amount), 0)
            _, contour_img = cv2.threshold(contour_img, 240, 255, cv2.THRESH_BINARY)
        
        # Passo 4: Adicionar detalhes internos importantes
        # Detectar bordas internas com Canny mais sensível
        canny_low2 = int(10 * det_factor)
        canny_high2 = int(80 * det_factor)
        internal_edges = cv2.Canny(gray, canny_low2, canny_high2)
        
        # Diluir bordas internas para minimizar ruído
        kernel = np.ones((2, 2), np.uint8)
        internal_edges = cv2.morphologyEx(internal_edges, cv2.MORPH_CLOSE, kernel)
        
        # Combinar bordas externas com internas (com menos peso nas internas)
        result = cv2.addWeighted(contour_img, 0.7, cv2.bitwise_not(internal_edges), 0.3, 0)
        
        # Binarizar novamente para garantir preto e branco puro
        _, result = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)
        
        return result

    def estilo_cartoon_contours(self, img):
        """Estilo cartoon simplificado com contornos mais definidos"""
        # Reduzir ruído com preservação de bordas
        color = cv2.bilateralFilter(img, d=9, sigmaColor=300, sigmaSpace=300)
        
        # Converter para tons de cinza
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordas com Canny
        detalhe = self.detalhe_var.get()
        if detalhe == "Baixo":
            low_thresh, high_thresh = 20, 100
        elif detalhe == "Alto":
            low_thresh, high_thresh = 50, 150
        else:  # Médio
            low_thresh, high_thresh = 30, 130
            
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        
        # Dilatar bordas para contornos mais fortes
        espessura = self.espessura_var.get()
        if espessura == "Fino":
            kernel_size = 1
        elif espessura == "Grosso":
            kernel_size = 3
        else:  # Médio
            kernel_size = 2
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Inverter cores para contornos pretos em fundo branco
        edges = cv2.bitwise_not(edges)
        
        # Aplicar suavização se necessário
        suavidade = self.suavidade_var.get() / 100.0
        if suavidade > 0.2:
            blur_amount = int(3 * suavidade)
            if blur_amount % 2 == 0:
                blur_amount += 1
            edges = cv2.GaussianBlur(edges, (blur_amount, blur_amount), 0)
            _, edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
        
        return edges

    def estilo_contornos_limpos(self, img):
        """Contornos limpos e bem definidos"""
        # Pré-processamento para reduzir ruído
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        
        # Ajustar parâmetros baseado no nível de detalhe
        detalhe = self.detalhe_var.get()
        if detalhe == "Baixo":
            adaptiveBlockSize = 11
            adaptiveC = 9
        elif detalhe == "Alto": 
            adaptiveBlockSize = 7
            adaptiveC = 3
        else:  # Médio
            adaptiveBlockSize = 9
            adaptiveC = 5
            
        # Usar threshold adaptativo para encontrar bordas
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, adaptiveBlockSize, adaptiveC)
        
        # Encontrar e desenhar contornos para ter mais controle
        contours, hierarchy = cv2.findContours(255 - edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        min_area = 10  # ajustar conforme necessário
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Desenhar contornos em imagem branca
        contour_img = np.ones_like(gray) * 255
        
        espessura = self.espessura_var.get()
        if espessura == "Fino":
            thickness = 1
        elif espessura == "Grosso":
            thickness = 3
        else:  # Médio
            thickness = 2
            
        cv2.drawContours(contour_img, filtered_contours, -1, (0), thickness)
        
        # Aplicar suavização conforme necessário
        suavidade = self.suavidade_var.get() / 100.0
        if suavidade > 0.1:
            blur_amount = int(5 * suavidade)
            if blur_amount % 2 == 0:
                blur_amount += 1
            contour_img = cv2.GaussianBlur(contour_img, (blur_amount, blur_amount), 0)
            _, contour_img = cv2.threshold(contour_img, 200, 255, cv2.THRESH_BINARY)
        
        return contour_img

    def estilo_face_focus(self, img):
    """Estilo que foca em detectar e destacar faces"""
    # Converter para RGB para MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inicializar detectores
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5) as face_detection:
        
        # Detectar faces
        results = face_detection.process(img_rgb)
        
        # Preparar imagem para desenho
        height, width = img.shape[:2]
        output_img = np.ones((height, width), dtype=np.uint8) * 255
        
        if results.detections:
            # Processar cada face detectada
            for detection in results.detections:
                # Obter caixa delimitadora
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Extrair região da face
                face_roi = img[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
                
                if face_roi.size > 0:
                    # Processar a face com detalhamento especial
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Ajustar parâmetros com base no nível de detalhe
                    detalhe = self.detalhe_var.get()
                    if detalhe == "Baixo":
                        threshold1, threshold2 = 50, 150
                    elif detalhe == "Alto":
                        threshold1, threshold2 = 30, 120
                    else:  # Médio
                        threshold1, threshold2 = 40, 130
                    
                    # Detectar bordas na face
                    face_edges = cv2.Canny(face_gray, threshold1, threshold2)
                    
                    # Dilatar bordas com base na espessura
                    espessura = self.espessura_var.get()
                    if espessura == "Fino":
                        kernel_size = 1
                    elif espessura == "Grosso":
                        kernel_size = 3
                    else:  # Médio
                        kernel_size = 2
                        
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    face_edges = cv2.dilate(face_edges, kernel, iterations=1)
                    
                    # Inverter para ter linhas pretas
                    face_edges = cv2.bitwise_not(face_edges)
                    
                    # Copiar as bordas da face para a imagem de saída
                    output_img[max(0, y):min(height, y + h), max(0, x):min(width, x + w)] = face_edges
        
        # Para o restante da imagem, aplicar um processamento mais leve
        if results.detections:
            # Criar máscara de faces
            face_mask = np.zeros((height, width), dtype=np.uint8)
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Expandir um pouco a área da face para incluir o contorno
                expansion = int(min(w, h) * 0.1)
                x = max(0, x - expansion)
                y = max(0, y - expansion)
                w = min(width - x, w + 2 * expansion)
                h = min(height - y, h + 2 * expansion)
                
                cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
            
            # Inverter máscara para obter área de fundo
            bg_mask = cv2.bitwise_not(face_mask)
            
            # Processar fundo de forma mais simples
            bg_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
            
            # Canny com parâmetros menos sensíveis para o fundo
            bg_edges = cv2.Canny(bg_blur, 100, 200)
            bg_edges = cv2.dilate(bg_edges, np.ones((2, 2), np.uint8), iterations=1)
            bg_edges = cv2.bitwise_not(bg_edges)
            
            # Aplicar apenas nas áreas de fundo
            bg_result = cv2.bitwise_and(bg_edges, bg_edges, mask=bg_mask)
            
            # Combinar faces processadas com fundo
            output_img = cv2.bitwise_and(output_img, output_img, mask=face_mask)
            output_img = cv2.add(output_img, bg_result)
        else:
            # Se não houver faces, aplicar estilo de lineart simples
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            output_img = cv2.bitwise_not(edges)
    
        # Aplicar suavização conforme necessário
        suavidade = self.suavidade_var.get() / 100.0
        if suavidade > 0.1:
            blur_amount = int(5 * suavidade)
            if blur_amount % 2 == 0:
                blur_amount += 1
            output_img = cv2.GaussianBlur(output_img, (blur_amount, blur_amount), 0)
            _, output_img = cv2.threshold(output_img, 200, 255, cv2.THRESH_BINARY)
    
    return output_img

def processar_ia_replicate(self, image_path):
    """Processa uma imagem usando o modelo LineArt da Replicate"""
    if not REPLICATE_API_TOKEN:
        messagebox.showerror("Erro", "Token da API Replicate não configurado. Verifique seu arquivo .env")
        return None
        
        self.status_var.set("Enviando para a API do Replicate... (pode demorar)")
        self.root.update()
    
    try:
        # Usar o modelo LineArt do Replicate
        output = replicate.run(
            "jagilley/controlnet-scribble:435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117",
            input={
                "image": open(image_path, "rb"),
                "prompt": "line art drawing, clean lines, coloring book page, black and white",
                "num_samples": "1",
                "image_resolution": "512",
                "n_prompt": "color, shading, grayscale, texture, noise",
            }
        )
        
        if output and isinstance(output, list) and len(output) > 0:
            # Baixar a imagem resultante
            response = requests.get(output[0])
            if response.status_code == 200:
                # Converter de bytes para imagem
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                # Binarizar para preto e branco
                _, img_bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
                
                return img_bw
            else:
                raise Exception(f"Erro ao baixar resultado: {response.status_code}")
        else:
            raise Exception("Resposta vazia da API")
            
    except Exception as e:
        messagebox.showerror("Erro", f"Erro na API Replicate: {str(e)}")
        self.status_var.set(f"Erro na API: {str(e)}")
        return None

def processar_ia_huggingface(self, image_path):
    """Processa uma imagem usando o modelo ControlNet da Hugging Face"""
    if not HUGGINGFACE_API_TOKEN:
        messagebox.showerror("Erro", "Token da API Hugging Face não configurado. Verifique seu arquivo .env")
        return None
        
    self.status_var.set("Enviando para a API do Hugging Face... (pode demorar)")
    self.root.update()
    
    try:
        # URL da API ControlNet para geração de linha
        API_URL = "https://api-inference.huggingface.co/models/lllyasviel/control_v11p_sd15_scribble"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        # Preparar imagem para envio
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        # Criar payload com parâmetros
        payload = {
            "inputs": image_bytes,
            "options": {
                "wait_for_model": True,
                "use_cache": False,
                "prompt": "line art drawing, clean lines, coloring book page, black and white",
                "negative_prompt": "color, shading, grayscale, texture, noise"
            }
        }
        
        # Enviar requisição
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        
        if response.status_code == 200:
            # Converter de bytes para imagem
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            # Binarizar para preto e branco
            _, img_bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            
            return img_bw
        else:
            raise Exception(f"Erro na API: {response.status_code}, {response.text}")
            
    except Exception as e:
        messagebox.showerror("Erro", f"Erro na API Hugging Face: {str(e)}")
        self.status_var.set(f"Erro na API: {str(e)}")
        return None

if __name__ == "__main__":
    # Criar a janela principal
    root = tk.Tk()
    app = LivroColorirApp(root)
    root.mainloop()