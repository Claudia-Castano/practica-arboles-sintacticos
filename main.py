
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QRadioButton
from nltk import CFG, ChartParser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Nodo:
    def __init__(self, etiqueta, hijos=None):
        self.etiqueta = etiqueta
        self.hijos = hijos if hijos else []


class Gramatica:
    def __init__(self, texto_gramatica):
        self.cfg = CFG.fromstring(texto_gramatica)
        self.parser = ChartParser(self.cfg)

    def parsear(self, expresion):
        tokens = expresion.strip().split()
        return list(self.parser.parse(tokens))


class Derivador:
    def __init__(self, arbol_nltk):
        self.arbol = arbol_nltk

    def obtener_pasos(self, tipo='izquierda'):
        pasos = []
        self._derivar(self.arbol, pasos, tipo)
        return pasos

    def _derivar(self, arbol, pasos, tipo):
        produccion = f"{arbol.label()} -> {' '.join(str(h) if isinstance(h, str) else h.label() for h in arbol)}"
        pasos.append(produccion)
        hijos = list(arbol)
        if tipo == 'derecha':
            hijos = list(reversed(hijos))
        for hijo in hijos:
            if not isinstance(hijo, str):
                self._derivar(hijo, pasos, tipo)


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generador de Árboles Sintácticos")
        self.setMinimumSize(1200, 700)
        self.construir_interfaz()

    def construir_interfaz(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)

        panel_izq = QWidget()
        layout_izq = QVBoxLayout(panel_izq)

        layout_izq.addWidget(QLabel("Gramática (formato BNF):"))
        self.texto_gramatica = QTextEdit()
        self.texto_gramatica.setPlaceholderText("Ej:\nE -> E '+' T | T\nT -> T '*' F | F\nF -> '(' E ')' | 'a' | 'b'")
        self.texto_gramatica.setMaximumHeight(200)
        layout_izq.addWidget(self.texto_gramatica)

        layout_izq.addWidget(QLabel("Expresión a derivar:"))
        self.texto_expresion = QTextEdit()
        self.texto_expresion.setPlaceholderText("Ej: a + b")
        self.texto_expresion.setMaximumHeight(60)
        layout_izq.addWidget(self.texto_expresion)

        layout_izq.addWidget(QLabel("Tipo de derivación:"))
        self.radio_izquierda = QRadioButton("Izquierda")
        self.radio_derecha = QRadioButton("Derecha")
        self.radio_izquierda.setChecked(True)
        layout_izq.addWidget(self.radio_izquierda)
        layout_izq.addWidget(self.radio_derecha)

        btn_generar = QPushButton("Generar Derivación")
        btn_generar.clicked.connect(self.generar)
        layout_izq.addWidget(btn_generar)

        layout_izq.addWidget(QLabel("Derivación paso a paso:"))
        self.texto_derivacion = QTextEdit()
        self.texto_derivacion.setReadOnly(True)
        layout_izq.addWidget(self.texto_derivacion)

        panel_der = QWidget()
        layout_der = QVBoxLayout(panel_der)

        layout_der.addWidget(QLabel("Árbol de Derivación:"))
        self.figura1, self.ax1 = plt.subplots(figsize=(7, 5))
        self.canvas1 = FigureCanvas(self.figura1)
        layout_der.addWidget(self.canvas1)

        layout_der.addWidget(QLabel("Árbol de Sintaxis Abstracta (AST):"))
        self.figura2, self.ax2 = plt.subplots(figsize=(5, 4))
        self.canvas2 = FigureCanvas(self.figura2)
        layout_der.addWidget(self.canvas2)

        layout_principal.addWidget(panel_izq, 1)
        layout_principal.addWidget(panel_der, 2)

    def generar(self):
        try:
            texto_g = self.texto_gramatica.toPlainText().strip()
            expresion = self.texto_expresion.toPlainText().strip()
            if not texto_g or not expresion:
                self.texto_derivacion.setText("Por favor ingresa la gramática y la expresión.")
                return
            gramatica = Gramatica(texto_g)
            arboles = gramatica.parsear(expresion)
            if not arboles:
                self.texto_derivacion.setText("No se pudo derivar la expresión con esta gramática.")
                return
            arbol = arboles[0]
            tipo = 'izquierda' if self.radio_izquierda.isChecked() else 'derecha'
            pasos = Derivador(arbol).obtener_pasos(tipo)
            self.texto_derivacion.setText("\n".join(pasos))
            self.dibujar_arbol(arbol)
            self.dibujar_ast(arbol)
        except Exception as e:
            self.texto_derivacion.setText(f"Error: {str(e)}")

    # ── Árbol de derivación ──

    def _contar_hojas(self, arbol):
        hijos = list(arbol)
        if all(isinstance(h, str) for h in hijos):
            return max(1, len(hijos))
        total = 0
        for h in hijos:
            if isinstance(h, str):
                total += 1
            else:
                total += self._contar_hojas(h)
        return total

    def _calcular_pos(self, arbol, posiciones, contador, x, ancho, y):
        uid = contador[0]
        contador[0] += 1
        arbol._uid = uid
        posiciones[uid] = (x, -y)

        if not hasattr(arbol, '_hijos_uid'):
            arbol._hijos_uid = {}

        hijos = list(arbol)
        pesos = []
        for h in hijos:
            if isinstance(h, str):
                pesos.append(1)
            else:
                pesos.append(self._contar_hojas(h))
        total = sum(pesos)

        cursor = x - ancho / 2
        for i, hijo in enumerate(hijos):
            fraccion = pesos[i] / total
            hijo_ancho = ancho * fraccion
            hijo_x = cursor + hijo_ancho / 2
            cursor += hijo_ancho
            if isinstance(hijo, str):
                uid_hijo = contador[0]
                contador[0] += 1
                arbol._hijos_uid[i] = uid_hijo
                posiciones[uid_hijo] = (hijo_x, -(y + 1))
            else:
                self._calcular_pos(hijo, posiciones, contador, hijo_x, hijo_ancho, y + 1)

    def _dibujar(self, arbol, posiciones, ax, padre_pos):
        uid = getattr(arbol, '_uid', None)
        pos = posiciones.get(uid)
        if pos is None:
            return
        ax.plot(pos[0], pos[1], 'o', markersize=22, color='#AFA9EC', zorder=3)
        ax.text(pos[0], pos[1], arbol.label(), ha='center', va='center', fontsize=9, zorder=4)
        if padre_pos:
            ax.plot([padre_pos[0], pos[0]], [padre_pos[1], pos[1]], 'k-', linewidth=0.8, zorder=2)
        hijos_uid = getattr(arbol, '_hijos_uid', {})
        for i, hijo in enumerate(arbol):
            if isinstance(hijo, str):
                hijo_pos = posiciones.get(hijos_uid.get(i))
                if hijo_pos:
                    ax.plot(hijo_pos[0], hijo_pos[1], 's', markersize=20, color='#f9e2af', zorder=3)
                    ax.text(hijo_pos[0], hijo_pos[1], hijo, ha='center', va='center', fontsize=9, zorder=4)
                    ax.plot([pos[0], hijo_pos[0]], [pos[1], hijo_pos[1]], 'k-', linewidth=0.8, zorder=2)
            else:
                self._dibujar(hijo, posiciones, ax, pos)

    def dibujar_arbol(self, arbol_nltk):
        self.ax1.clear()
        posiciones = {}
        contador = [0]
        hojas = self._contar_hojas(arbol_nltk)
        self._calcular_pos(arbol_nltk, posiciones, contador, x=0, ancho=hojas * 1.8, y=0)
        self._dibujar(arbol_nltk, posiciones, self.ax1, padre_pos=None)
        self.ax1.set_title("Árbol de Derivación")
        self.ax1.axis('off')
        self.canvas1.draw()

    # ── AST ──

    def _construir_ast(self, arbol):
        hijos_reales = [h for h in arbol if not isinstance(h, str)]
        hojas = [h for h in arbol if isinstance(h, str)]

        # Nodo hoja terminal
        if len(hijos_reales) == 0:
            # Ignorar paréntesis, devolver el valor
            valores = [h for h in hojas if h not in ('(', ')')]
            if valores:
                return Nodo(valores[0])
            return None

        # Nodo con un solo hijo no terminal y sin operador -> bajar
        operadores = [h for h in hojas if h in ('+', '-', '*', '/')]
        if not operadores and len(hijos_reales) == 1:
            return self._construir_ast(hijos_reales[0])

        # Nodo con paréntesis: F -> ( E ) -> bajar directo al hijo
        if '(' in hojas and ')' in hojas and len(hijos_reales) == 1:
            return self._construir_ast(hijos_reales[0])

        # Nodo con operador -> operador es la raíz
        if operadores:
            hijos_ast = [self._construir_ast(h) for h in hijos_reales]
            hijos_ast = [h for h in hijos_ast if h is not None]
            return Nodo(operadores[0], hijos_ast)

        # Caso general
        hijos_ast = [self._construir_ast(h) for h in hijos_reales]
        hijos_ast = [h for h in hijos_ast if h is not None]
        return Nodo(arbol.label(), hijos_ast)

    def _calcular_pos_ast(self, nodo, posiciones, x, ancho, y):
        posiciones[id(nodo)] = (x, -y)
        n = len(nodo.hijos)
        if n == 0:
            return
        cursor = x - ancho / 2
        peso = ancho / n
        for hijo in nodo.hijos:
            hijo_x = cursor + peso / 2
            cursor += peso
            self._calcular_pos_ast(hijo, posiciones, hijo_x, peso, y + 1)

    def _dibujar_ast(self, nodo, posiciones, ax, padre_pos):
        pos = posiciones.get(id(nodo))
        if pos is None:
            return
        ax.plot(pos[0], pos[1], 'o', markersize=22, color='#5DCAA5', zorder=3)
        ax.text(pos[0], pos[1], nodo.etiqueta, ha='center', va='center', fontsize=9, zorder=4)
        if padre_pos:
            ax.plot([padre_pos[0], pos[0]], [padre_pos[1], pos[1]], 'k-', linewidth=0.8, zorder=2)
        for hijo in nodo.hijos:
            self._dibujar_ast(hijo, posiciones, ax, pos)

    def dibujar_ast(self, arbol_nltk):
        self.ax2.clear()
        nodo = self._construir_ast(arbol_nltk)
        if nodo is None:
            return
        posiciones = {}
        hojas = self._contar_hojas_ast(nodo)
        self._calcular_pos_ast(nodo, posiciones, x=0, ancho=hojas * 1.2, y=0)
        self._dibujar_ast(nodo, posiciones, self.ax2, padre_pos=None)
        self.ax2.set_title("AST")
        self.ax2.axis('off')
        self.canvas2.draw()

    def _contar_hojas_ast(self, nodo):
        if not nodo.hijos:
            return 1
        return sum(self._contar_hojas_ast(h) for h in nodo.hijos)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())

