import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import threading

# These are used for generic declarations, similar
# to #defines in c/c++. Since this is python we can
# change 'em midway too, if needed. Huh.
GUI_SMPLCNT = 25                # Sample count
GUI_F1 = 3                      # Frequency 1
GUI_F2 = 7                      # Frequency 1
GUI_AMP = 60                    # Signal Amplitude

GUI_NRZ = "NZR-Polar"
GUI_MAN = "Manchster"
GUI_BIP = "Bipolar"

GUI_ASK = "ASK"
GUI_FSK = "FSK"
GUI_BPSK = "BPSK"
GUI_QPSK = "QPSK"
GUI_16QAM = "16QAM"

GUI_CONTAGEM = "Contagem de Caracteres"
GUI_FLAGBYTE = "FLAG e byte stuffing"
GUI_FLAGBIT = "FLAG e bit stuffing"

GUI_PBIT = "Bit de Paridade"
GUI_CHKSM = "Checksum"
GUI_CRC32 = "CRC-32"
GUI_HAM = "Hamming (com correção)"

class Programa(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Simulador de Camadas Física e Enlace")
        # self.set_default_size(1600, 900)
        
        # Basic layout
        self.grid = Gtk.Grid()
        self.grid.set_column_spacing(15)
        self.grid.set_row_spacing(10)
        self.grid.set_border_width(10)
        self.add(self.grid)
        
        # Basic input entries for the desired simulation
        self.input_section()
        self.digmod_section()
        self.algmod_section()
        self.framing_section()
        self.errordetct_section()
        self.errorcurve_section()

        # Run() button
        self.run_button = Gtk.Button(label="Executar Simulação")
        self.run_button.connect("clicked", self.on_run_clicked)
        self.grid.attach(self.run_button, 12, 0, 1, 2)
        
    def input_section(self):
        # Message input area
        input_label = Gtk.Label(label="v Mensagem a Enviar v")
        self.grid.attach(input_label, 0, 0, 2, 1)
        self.input_message = Gtk.Entry()
        self.input_message.set_placeholder_text("Digite aqui!")
        self.grid.attach(self.input_message, 0, 1, 2, 1)
        
    def digmod_section(self):
        # Digital modulation options
        digital_label = Gtk.Label(label="Modulação Digital:")
        self.grid.attach(digital_label, 3, 0, 1, 1)
        
        self.digmod_option = Gtk.ComboBoxText()
        self.digmod_option.append_text(GUI_NRZ)
        self.digmod_option.append_text(GUI_MAN)
        self.digmod_option.append_text(GUI_BIP)
        self.digmod_option.set_active(0)
        self.grid.attach(self.digmod_option, 4, 0, 2, 1)
        
    def algmod_section(self):
        # Analog modulation options
        analog_label = Gtk.Label(label="Modulação por Portadora:")
        self.grid.attach(analog_label, 3, 1, 1, 1)
        
        self.algmod_option = Gtk.ComboBoxText()
        self.algmod_option.append_text(GUI_ASK)
        self.algmod_option.append_text(GUI_FSK)
        self.algmod_option.append_text(GUI_BPSK)
        self.algmod_option.append_text(GUI_QPSK)
        self.algmod_option.append_text(GUI_16QAM)
        self.algmod_option.set_active(0)
        self.grid.attach(self.algmod_option, 4, 1, 2, 1)
        
    def framing_section(self):
        # Framing options
        framing_label = Gtk.Label(label="Enquadramento:")
        self.grid.attach(framing_label, 6, 0, 1, 1)
        
        self.framing_option = Gtk.ComboBoxText()
        self.framing_option.append_text(GUI_CONTAGEM)
        self.framing_option.append_text(GUI_FLAGBYTE)
        self.framing_option.append_text(GUI_FLAGBIT)
        self.framing_option.set_active(0)
        self.grid.attach(self.framing_option, 7, 0, 2, 1)
    
    def errordetct_section(self):
        # Error detection options
        error_label = Gtk.Label(label="Detecção de Erros:")
        self.grid.attach(error_label, 6, 1, 1, 1)

        self.errordetct_option = Gtk.ComboBoxText()
        self.errordetct_option.append_text(GUI_PBIT)
        self.errordetct_option.append_text(GUI_CHKSM)
        self.errordetct_option.append_text(GUI_CRC32)
        self.errordetct_option.append_text(GUI_HAM)
        self.errordetct_option.set_active(0)
        self.grid.attach(self.errordetct_option, 7, 1, 2, 1)
        
    def errorcurve_section(self):
        # Error curve parameters
        errorcurve_label = Gtk.Label(label="----->\nRuído\n----->")
        self.grid.attach(errorcurve_label, 9, 0, 1, 2)

        self.error_mean = Gtk.Entry()
        self.error_mean.set_placeholder_text("Média (μ = 0)")
        self.grid.attach(self.error_mean, 10, 0, 2, 1)

        self.error_variance = Gtk.Entry()
        self.error_variance.set_placeholder_text("Variância (σ² = 1)")
        self.grid.attach(self.error_variance, 10, 1, 2, 1)

    def on_run_clicked(self, button):
        # Get all configurations and verify their validity
        message = self.input_message.get_text()
        if not message.isascii() or message == "":
            print('Invalid message! Must be ASCII!')
            return

        digital_mod: str = self.digmod_option.get_active_text()
        analog_mod: str = self.algmod_option.get_active_text()
        framing_method: str = self.framing_option.get_active_text()
        error_method: str = self.errordetct_option.get_active_text()

        errorcurve_mean: float = self.error_mean.get_text()
        try:
            if errorcurve_mean == "":
                errorcurve_mean = 0
            else:
                errorcurve_mean = float(errorcurve_mean)
        except ValueError:
            print("Mean must be a float!")
            return

        errorcurve_variance: float = self.error_variance.get_text()
        try:
            if errorcurve_variance == "":
                errorcurve_variance = 1.0
            else:
                errorcurve_variance = float(errorcurve_variance)
            
            if errorcurve_variance < 0:
                raise ValueError
        except ValueError:
            print("Variance must be a positive float!")
            return

        if not self.run(message,
                        digital_mod,
                        analog_mod,
                        framing_method,
                        error_method,
                        errorcurve_mean,
                        errorcurve_variance):
            print("Something went TERRIBLY wrong")
        
    def run(self,
            message, 
            digital_mod, 
            analog_mod,
            framing_method,
            error_method,
            errorcurve_mean,
            errorcurve_variance) -> bool:
        import CamadaEnlace as ce
        import CamadaFisica as cf
        import Receptor     as rp
        import Transmissor  as tm
        import utils
        import time

        dmsg: list[bool] = utils.string_to_bitstream(message)
        if dmsg is None:
            return False

        # utils.logmsg(str(dmsg))
        
        # Digital encoding
        dmmsg: npt.NDArray = np.empty(0)
        if digital_mod == GUI_NRZ:
            dmmsg = cf.nrz_polar(dmsg)
        elif digital_mod == GUI_MAN:
            dmmsg = cf.manchester(dmsg)
        elif digital_mod == GUI_BIP:
            dmmsg = cf.bipolar(dmsg)
        else:
            return False
        if dmmsg is None:
            return False
        self.digital_plot(dmmsg, "Modulação digital")


        # Analog modulation of only the dsignal, no noise, no framing
        auxmsg: npt.NDArray = np.empty(0)
        if analog_mod == GUI_ASK:
            auxmsg = cf.ASK_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_FSK:
            auxmsg = cf.FSK_modulation(dmsg, (GUI_F1, GUI_F2), smplcnt=GUI_SMPLCNT, amp=GUI_AMP)
        elif analog_mod == GUI_BPSK:
            auxmsg = cf.BPSK_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_QPSK:
            auxmsg = cf.QPSK_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_16QAM:
            auxmsg = cf.QAM_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        else:
            return False
        self.analog_plot(auxmsg, "Apenas mensagem, sem enlace ou ruído")


        # Framing type
        if framing_method == GUI_CONTAGEM:
            enlace_type = ce.FP.CHAR_COUNT
        elif framing_method == GUI_FLAGBYTE:
            enlace_type = ce.FP.BIT_ORIENTED_FLAGGING
        elif framing_method == GUI_FLAGBIT:
            enlace_type = ce.FP.BYTE_ORIENTED_FLAGGING
        else:
            return False

        # Error detection / correction type
        if error_method == GUI_PBIT:
            error_type = ce.EDP.PARITY_BIT
        elif error_method == GUI_CHKSM:
            error_type = ce.EDP.CHECKSUM
        elif error_method == GUI_CRC32:
            error_type = ce.EDP.CRC32
        elif error_method == GUI_HAM:
            error_type = ce.EDP.HAMMING
        else:
            return False


        # Apply Framing Protocol + Error Detection Protocol
        PAYLOADSIZE = 32
        dmsg = ce.split_bitstream_into_payloads(dmsg, PAYLOADSIZE)
        if dmsg is None: return False
        dmsg = ce.add_padding_and_padding_size(dmsg, PAYLOADSIZE)
        if dmsg is None: return False
        dmsg = ce.add_EDC(dmsg, error_type)
        if dmsg is None: return False
        dmsg = ce.add_framing_protocol(dmsg, enlace_type)
        if dmsg is None: return False
        dmsg = ce.list_linearize(dmsg)
        if dmsg is None: return False

        # Creates and starts receiver at a different thread
        # This is necessary for the asynchronous socket implementation
        receiver = rp.Receptor()
        receiver_trd = threading.Thread(target=receiver.start_server, daemon=True)
        receiver_trd.start()
        time.sleep(0.5)

        # Encode message
        amsg: npt.NDArray = np.empty(0)
        if analog_mod == GUI_ASK:
            amsg = cf.ASK_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_FSK:
            amsg = cf.FSK_modulation(dmsg, (GUI_F1, GUI_F2), smplcnt=GUI_SMPLCNT, amp=GUI_AMP)
        elif analog_mod == GUI_BPSK:
            amsg = cf.BPSK_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_QPSK:
            amsg = cf.QPSK_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_16QAM:
            amsg = cf.QAM_modulation(dmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        else:
            return False
        if amsg is None:
            return False
        
        # Add noise
        amsg = utils.samples_addnoise(amsg, average=errorcurve_mean, spread=errorcurve_variance)
        if amsg is None:
            return False

        # Send message
        if not tm.send_to_server(amsg):
            return False
        
        # Receive message
        receiver.flag_ready.wait(timeout=5)
        rmsg: npt.NDArray = receiver.interpreted_data
        if rmsg is None:
            return False

        # Display received signal
        self.analog_plot(rmsg, "Sinal recebido enlaçado")
        
        # Decode message
        if analog_mod == GUI_ASK:
            rmsg = cf.ASK_demodulation(rmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_FSK:
            rmsg = cf.FSK_demodulation(rmsg, (GUI_F1, GUI_F2), smplcnt=GUI_SMPLCNT, amp=GUI_AMP)
        elif analog_mod == GUI_BPSK:
            rmsg = cf.BPSK_demodulation(rmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_QPSK:
            rmsg = cf.QPSK_demodulation(rmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        elif analog_mod == GUI_16QAM:
            rmsg = cf.QAM_demodulation(rmsg, smplcnt=GUI_SMPLCNT, f=GUI_F1, amp=GUI_AMP)
        else:
            return False
        if rmsg is None:
            return False

        # Deframe message
        rmsg = ce.remove_framing_protocol(rmsg, enlace_type)
        if rmsg is None: return False
            # Check for errors (and correct them if it is the case)
        rmsg = ce.remove_EDC(rmsg, error_type)
        if rmsg is None: return False
        rmsg = ce.remove_paddings(rmsg)
        if rmsg is None: return False
        rmsg = ce.list_linearize(rmsg)
        if rmsg is None: return False

        # Display analyzed bitstream
        self.digital_plot(rmsg, "Mensagem decodificada")

        # Display message
        print(utils.bitstream_to_string(rmsg))
        
        # Everything went right!
        return True

    def analog_plot(self, signal, title):
        mod = self.algmod_option.get_active_text()
        graph = plt.Figure(figsize=(5, 3), dpi=100)
        ax = graph.add_subplot(111)
        ax.plot(signal)
        ax.set_title(f"Sinal Analógico")
        ax.set_xlabel("Amostras")
        ax.set_ylabel("Leitura de tensão")

        canvas = FigureCanvas(graph)
        canvas.set_size_request(800, 400)
        win = Gtk.Window(title=f"Analógico - {title}")
        win.add(canvas)
        win.show_all()

    def digital_plot(self, dsignal, title):
        graph = plt.Figure(figsize=(5, 3), dpi=100)
        ax = graph.add_subplot(111)

        mod = self.digmod_option.get_active_text()
        
        if title == GUI_MAN:
            amnt_smpl = len(dsignal) // 2
        else:
            amnt_smpl = len(dsignal)

        x = np.linspace(0, amnt_smpl, amnt_smpl, endpoint=False)

        ax.plot(x, dsignal, drawstyle="steps-pre")
        ax.set_title(f"Sinal Digital")
        ax.set_xlabel("Amostras")
        ax.set_ylabel("Leitura de tensão")
        ax.set_xticks(np.arange(0, amnt_smpl + 1))
        ax.set_xlim(0, amnt_smpl)

        canvas = FigureCanvas(graph)
        canvas.set_size_request(800, 400)
        win = Gtk.Window(title=f"Digital - {title}")
        win.add(canvas)
        win.show_all()


if __name__ == "__main__":
    win = Programa()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()