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
GUI_F2 = 7                      # Frequency 2
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

GUI_PAYLOADSIZE = 32

class Programa(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Simre;Camfe")
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
        self.extraparam_smpls()

        # Run() button
        self.run_button = Gtk.Button(label="Executar Simulação")
        self.run_button.connect("clicked", self.on_run_clicked)
        self.grid.attach(self.run_button, 9, 0, 2, 2)
        
    def input_section(self):
        # Message input area
        self.input_message = Gtk.Entry()
        self.input_message.set_placeholder_text("Mensagem!")
        self.grid.attach(self.input_message, 0, 0, 2, 2)
        
    def digmod_section(self):
        # Digital modulation options
        digital_label = Gtk.Label(label="Modulação Digital ->")
        self.grid.attach(digital_label, 3, 0, 1, 1)
        
        self.digmod_option = Gtk.ComboBoxText()
        self.digmod_option.append_text(GUI_NRZ)
        self.digmod_option.append_text(GUI_MAN)
        self.digmod_option.append_text(GUI_BIP)
        self.digmod_option.set_active(0)
        self.grid.attach(self.digmod_option, 4, 0, 2, 1)
        
    def algmod_section(self):
        # Analog modulation options
        analog_label = Gtk.Label(label="Modulação por Portadora ->")
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
        framing_label = Gtk.Label(label="Enquadramento ->")
        self.grid.attach(framing_label, 6, 0, 1, 1)
        
        self.framing_option = Gtk.ComboBoxText()
        self.framing_option.append_text(GUI_CONTAGEM)
        self.framing_option.append_text(GUI_FLAGBYTE)
        self.framing_option.append_text(GUI_FLAGBIT)
        self.framing_option.set_active(0)
        self.grid.attach(self.framing_option, 7, 0, 2, 1)
    
    def errordetct_section(self):
        # Error detection options
        error_label = Gtk.Label(label="Detecção de Erros ->")
        self.grid.attach(error_label, 6, 1, 1, 1)

        self.errordetct_option = Gtk.ComboBoxText()
        self.errordetct_option.append_text(GUI_PBIT)
        self.errordetct_option.append_text(GUI_CHKSM)
        self.errordetct_option.append_text(GUI_CRC32)
        self.errordetct_option.set_active(0)
        self.grid.attach(self.errordetct_option, 7, 1, 2, 1)
        
    def errorcurve_section(self):
        # Error curve parameters
        errorcurve_label = Gtk.Label(label="----->\nRuído\n----->")
        self.grid.attach(errorcurve_label, 6, 2, 1, 2)

        self.error_mean = Gtk.Entry()
        self.error_mean.set_placeholder_text("Média (Volts=0)")
        self.grid.attach(self.error_mean, 7, 2, 2, 1)

        self.error_variance = Gtk.Entry()
        self.error_variance.set_placeholder_text("Variância (Volts=1)")
        self.grid.attach(self.error_variance, 7, 3, 2, 1)
        pass
        
    def extraparam_smpls(self):
        # Sampling parameters
        extra_label = Gtk.Label(label="--------------->\nAmostragem\n--------------->")
        self.grid.attach(extra_label, 0, 2, 2, 2)

        self.extprm_sample = Gtk.Entry()
        self.extprm_sample.set_placeholder_text("Samples (25)")
        self.grid.attach(self.extprm_sample, 2, 2, 2, 1)

        self.extprm_amp = Gtk.Entry()
        self.extprm_amp.set_placeholder_text("Amplitude (Volts=60)")
        self.grid.attach(self.extprm_amp, 2, 3, 2, 1)

        self.extprm_f1 = Gtk.Entry()
        self.extprm_f1.set_placeholder_text("f1 (Hz=3)")
        self.grid.attach(self.extprm_f1, 4, 2, 2, 1)
        self.extprm_f2 = Gtk.Entry()
        self.extprm_f2.set_placeholder_text("f2 (Hz=7)")
        self.grid.attach(self.extprm_f2, 4, 3, 2, 1)

        extra_label = Gtk.Label(label="Paridade ímpar")
        self.grid.attach(extra_label, 9, 2, 1, 1)
        self.extprm_podd = Gtk.CheckButton()
        self.grid.attach(self.extprm_podd, 10, 2, 1, 1)

        self.extprm_ps = Gtk.Entry()
        self.extprm_ps.set_placeholder_text("Payloadsize=32")
        self.grid.attach(self.extprm_ps, 9, 3, 2, 1)




    def on_run_clicked(self, button):
        # Get all configurations and verify their validity
        global GUI_AMP, GUI_SMPLCNT, GUI_F1, GUI_F2, GUI_PAYLOADSIZE
        message = self.input_message.get_text()
        if not message.isascii() or message == "":
            self.text_show('Mensagem inválida!\nDeve conter apenas ASCII.')
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
            self.text_show("Média deve ser um float!")
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
            self.text_show("Variância deve ser um float positivo!")
            return

        smpl_cnt = self.extprm_sample.get_text()
        try:
            if smpl_cnt == "":
                smpl_cnt = 25
            else:
                smpl_cnt = int(smpl_cnt)
            
            if smpl_cnt <= 0:
                raise ValueError
            GUI_SMPLCNT = smpl_cnt
        except ValueError:
            self.text_show("Quantidade de samples deve ser um inteiro positivo!")
            return

        amp_val = self.extprm_amp.get_text()
        try:
            if amp_val == "":
                amp_val = 60
            else:
                amp_val = float(amp_val)

            GUI_AMP = amp_val
        except ValueError:
            self.text_show("Amplitude deve ser um float!")
            return

        f1_val = self.extprm_f1.get_text()
        try:
            if f1_val == "":
                f1_val = 3
            else:
                f1_val = float(f1_val)
            
            if f1_val <= 0:
                raise ValueError
            GUI_F1 = f1_val
        except ValueError:
            self.text_show("Frequências devem ser floats positivos não-nulos!")
            return

        f2_val = self.extprm_f2.get_text()
        try:
            if f2_val == "":
                f2_val = 7
            else:
                f2_val = float(f2_val)
            
            if f2_val <= 0:
                raise ValueError
            GUI_F2 = f2_val
        except ValueError:
            self.text_show("Frequências devem ser floats positivos não-nulos!")
            return

        parity_status = self.extprm_podd.get_active()

        ps_val = self.extprm_ps.get_text()
        try:
            if ps_val == "":
                ps_val = 32
            else:
                ps_val = int(ps_val)
            
            if ps_val <= 0:
                raise ValueError
            if ps_val % 2 == 1:
                raise ValueError
            GUI_PAYLOADSIZE = ps_val
        except ValueError:
            self.text_show("Payloadsize deve ser um inteiro positivo divisível por 2!")
            return

        if not self.run(message,
                        digital_mod,
                        analog_mod,
                        framing_method,
                        error_method,
                        errorcurve_mean,
                        errorcurve_variance,
                        parity_status):
            self.text_show("Algo aconteceu de errado durante o processamento. Tarefa abortada.")
        
    def run(self,
            message, 
            digital_mod, 
            analog_mod,
            framing_method,
            error_method,
            errorcurve_mean,
            errorcurve_variance,
            parity_status) -> bool:
        global GUI_SMPLCNT, GUI_F1, GUI_F2, GUI_AMP, GUI_NRZ, GUI_MAN
        global GUI_BIP, GUI_ASK, GUI_FSK, GUI_BPSK, GUI_QPSK, GUI_16QAM
        global GUI_CONTAGEM, GUI_FLAGBYTE, GUI_FLAGBIT, GUI_PBIT, GUI_CHKSM
        global GUI_CRC32, GUI_PAYLOADSIZE
        
        import CamadaEnlace as ce
        import CamadaFisica as cf
        import Receptor     as rp
        import Transmissor  as tm
        import utils
        import time

        textdisp = ""

        textdisp += "========= MENSAGEM INICIAL ==========\n"
        textdisp += message
        textdisp += "\n\n"

        dmsg: list[bool] = utils.string_to_bitstream(message)
        if dmsg is None:
            return False

        textdisp += "========= BITSTREAM DA MENSAGEM ==========\n"
        textdisp += np.array2string(np.array(dmsg, dtype='bool'), max_line_width=70, threshold=len(dmsg)+1)
        textdisp += "\n\n"
        
        # Digital encoding
        dmmsg: np.NDArray = np.empty(0)
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

        textdisp += "========= MODULAÇÃO DIGITAL ==========\n"
        textdisp += np.array2string(np.array(dmmsg, dtype='bool'), max_line_width=70, threshold=len(dmmsg)+1)
        textdisp += "\n\n"

        # Analog modulation of only the dsignal, no noise, no framing
        auxmsg: np.NDArray = np.empty(0)
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

        # textdisp += "========= MODULAÇÃO ANALÓGICA MENSAGEM ==========\n"
        # textdisp += np.array2string(auxmsg, max_line_width=70, threshold=len(auxmsg)+1)
        # textdisp += "\n\n"

        # Framing type
        if framing_method == GUI_CONTAGEM:
            enlace_type = ce.FP.CHAR_COUNT
        elif framing_method == GUI_FLAGBYTE:
            enlace_type = ce.FP.BYTE_ORIENTED_FLAGGING
        elif framing_method == GUI_FLAGBIT:
            enlace_type = ce.FP.BIT_ORIENTED_FLAGGING
        else:
            return False

        # Error detection / correction type
        if error_method == GUI_PBIT:
            error_type = ce.EDP.PARITY_BIT
        elif error_method == GUI_CHKSM:
            error_type = ce.EDP.CHECKSUM
        elif error_method == GUI_CRC32:
            error_type = ce.EDP.CRC32
        else:
            return False


        # Apply Framing Protocol + Error Detection Protocol
        dmsg = ce.split_bitstream_into_payloads(dmsg, GUI_PAYLOADSIZE)
        if dmsg is None: return False
        dmsg = ce.add_padding_and_padding_size(dmsg)
        if dmsg is None: return False
        dmsg = ce.add_EDC(dmsg, error_type, odd=parity_status)
        if dmsg is None: return False
        dmsg = ce.add_ECC(dmsg)
        if dmsg is None: return False
        dmsg = ce.add_framing_protocol(dmsg, enlace_type)
        if dmsg is None: return False
        # Saves number of sent frames
        sent_frames_count = len(dmsg)
        dmsg = ce.list_linearize(dmsg)
        if dmsg is None: return False
        dmsg = ce.add_padding_for_4bit_alignment(dmsg)
        if dmsg is None: return False

        textdisp += "========= CAMADA DE ENLACE ==========\n"
        textdisp += np.array2string(np.array(dmsg, dtype='bool'), max_line_width=70, threshold=len(dmsg)+1)
        textdisp += "\n\n"

        # Creates and starts receiver at a different thread
        # This is necessary for the asynchronous socket implementation
        receiver = rp.Receptor()
        receiver_trd = threading.Thread(target=receiver.start_server, daemon=True)
        receiver_trd.start()
        time.sleep(0.5)

        # Encode message
        amsg: np.NDArray = np.empty(0)
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

        # textdisp += "========= MODULAÇÃO ANALÓGICA COMPLETA ==========\n"
        # textdisp += np.array2string(amsg, max_line_width=70, threshold=len(amsg)+1)
        # textdisp += "\n\n"
        
        # Add noise
        amsg = utils.samples_addnoise(amsg, average=errorcurve_mean, spread=errorcurve_variance)
        if amsg is None:
            return False

        # textdisp += "========= RUÍDO ==========\n"
        # textdisp += np.array2string(amsg, max_line_width=70, threshold=len(amsg)+1)
        # textdisp += "\n\n"

        # Send message
        if not tm.send_to_server(amsg):
            return False
        
        # Receive message
        receiver.flag_ready.wait(timeout=5)
        rmsg: np.NDArray = receiver.interpreted_data
        if rmsg is None:
            return False

        # textdisp += "========= SINAL RECEBIDO ==========\n"
        # textdisp += np.array2string(rmsg, max_line_width=70, threshold=len(rmsg)+1)
        # textdisp += "\n\n"

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

        textdisp += "========= SINAL DIGITALIZADO ==========\n"
        textdisp += np.array2string(np.array(rmsg, dtype='bool'), max_line_width=70, threshold=len(rmsg)+1)
        textdisp += "\n\n"

        #Remove 4bit padding from message
        rmsg = ce.remove_padding_for_4bit_alignment(rmsg)

        # Deframe message
        rmsg = ce.remove_framing_protocol(rmsg, enlace_type)
        if rmsg is None: return False
        # Correct errors identified by the ECC
        rmsg = ce.ECC_fix_corrupted_bits(rmsg)
        # Remove ECC
        rmsg = ce.remove_ECC(rmsg)
        # Find corrupted frames
        corrupted_frames = ce.find_corrupted_frames(rmsg, error_type, odd=parity_status)
        # Saves number of received frames
        received_frames_count = len(rmsg)
        #Verifies if last frame is corrupted
        last_frame_corrupted = corrupted_frames[-1] == len(rmsg) if corrupted_frames else 0
        # Remove EDC
        rmsg = ce.remove_EDC(rmsg, error_type)
        if rmsg is None: return False
        # Remove padding
        rmsg = ce.remove_paddings(rmsg, last_frame_corrupted)
        if rmsg is None: return False
        # Flatten final result
        rmsg = ce.list_linearize(rmsg)
        if rmsg is None: return False

        textdisp += "========= SINAL DECODIFICADO ==========\n"
        textdisp += np.array2string(np.array(rmsg, dtype='bool'), max_line_width=70, threshold=len(rmsg)+1)
        textdisp += "\n\n"

        # Display analyzed bitstream
        self.digital_plot(rmsg, "Mensagem decodificada")

        textdisp += "=========== QUADROS INFO =============\n"
        textdisp += "Número de quadros enviados: " + str(sent_frames_count) + "\n"
        textdisp += "Número de quadros recebidos: " + str(received_frames_count) + "\n"
        textdisp += f"Encontrados {len(corrupted_frames)} quadros corrompidos pelo EDC.\n"
        if corrupted_frames:
            textdisp += f"Quadros corrompidos: {corrupted_frames}"
        textdisp += "\n\n"

        # Display message
        textdisp += "========= MENSAGEM FINAL =============\n"
        # Guarantees message size is divisible by 8
        # Otherwise, it can't be converted to ascii
        rmsg += [False for _ in range((8 - (len(rmsg) % 8)) % 8)]
        textdisp += utils.bitstream_to_string(rmsg)
        textdisp += "\n\n"
        
        # Display text
        self.text_show(textdisp)

        # Everything went right!
        return True

    def analog_plot(self, signal, title):
        mod = self.algmod_option.get_active_text()
        graph = plt.Figure()
        ax = graph.add_subplot()
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
        graph = plt.Figure()
        ax = graph.add_subplot()

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

    def text_show(self, message):
        # Text display
        label = Gtk.Label(label=message)
        scrlb = Gtk.ScrolledWindow()
        scrlb.add(label)
        scrlb.set_size_request(600, 600)
        win = Gtk.Window(title=f"Output de texto")
        win.add(scrlb)
        win.show_all()


if __name__ == "__main__":
    win = Programa()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()