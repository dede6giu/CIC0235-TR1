import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import numpy as np
import threading

# These are used for generic declarations, similar
# to #defines in c/c++
GUI_NRZ = "NZR-Polar"
GUI_MAN = "Manchster"
GUI_BIP = "Bipolar"

GUI_ASK = "ASK"
GUI_FSK = "FSK"
GUI_BPSK = "BPSK"
GUI_QPSK = "QPSK"
GUI_Q8AM = "Q8AM"

GUI_CONTAGEM = "Contagem de Caracteres"
GUI_FLAGBYTE = "FLAG e byte stuffing"
GUI_FLAGBIT = "FLAG e bit stuffing"

GUI_PBIT = "Bit de Paridade"
GUI_CRC32 = "CRC-32"
GUI_HAM = "Hamming (com correção)"

class Programa(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Simulador de Camadas Física e Enlace")
        self.set_default_size(1600, 900)
        
        # Basic layout
        self.grid = Gtk.Grid()
        self.grid.set_column_spacing(15)
        self.grid.set_row_spacing(10)
        self.add(self.grid)
        
        # Basic input entries for the desired simulation
        self.input_section()
        self.digmod_section()
        self.algmod_section()
        self.framing_section()
        self.errorcurve_section()
        
        # Graph space
        # TODO

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
        self.algmod_option.append_text(GUI_Q8AM)
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
        
        # Error detection options
        error_label = Gtk.Label(label="Detecção de Erros:")
        self.grid.attach(error_label, 6, 1, 1, 1)

        self.errordetct_option = Gtk.ComboBoxText()
        self.errordetct_option.append_text(GUI_PBIT)
        self.errordetct_option.append_text(GUI_CRC32)
        self.errordetct_option.append_text(GUI_HAM)
        self.errordetct_option.set_active(0)
        self.grid.attach(self.errordetct_option, 7, 1, 2, 1)
        
    def errorcurve_section(self):
        # Error curve parameters
        errorcurve_label = Gtk.Label(label="----->\nErros\n----->")
        self.grid.attach(errorcurve_label, 9, 0, 1, 2)

        self.error_mean = Gtk.Entry()
        self.error_mean.set_placeholder_text("Média (μ)")
        self.grid.attach(self.error_mean, 10, 0, 2, 1)

        self.error_variance = Gtk.Entry()
        self.error_variance.set_placeholder_text("Variância (σ²)")
        self.grid.attach(self.error_variance, 10, 1, 2, 1)

    def on_run_clicked(self, button):

        # Get all configurations and verify their validity
        message = self.input_message.get_text()
        if not message.isascii():
            print('Invalid message! Must be ASCII!')
            return

        digital_mod = self.digmod_option.get_active_text()
        analog_mod = self.algmod_option.get_active_text()
        framing_method = self.framing_option.get_active_text()
        error_method = self.errordetct_option.get_active_text()

        errorcurve_mean = self.error_mean.get_text()
        try:
            errorcurve_mean = float(errorcurve_mean)
        except ValueError:
            print("Mean must be a float!")
            return

        errorcurve_variance = self.error_variance.get_text()
        try:
            errorcurve_variance = float(errorcurve_variance)
            if errorcurve_variance < 0:
                raise ValueError
        except ValueError:
            print("Variance must be a positive float!")
            return

        if not self.run(message, digital_mod, analog_mod, framing_method, error_method):
            print("big boo boo error")
        
    def run(self, message, digital_mod, analog_mod, framing_method, error_method) -> bool:
        import CamadaEnlace as ce
        import CamadaFisica as cf
        import Receptor     as rp
        import Transmissor  as tm
        import utils
        import time

        dmsg: list[bool] = utils.string_to_bitstream(message)
        if dmsg is None:
            print("Not ASCII encoded message!")
            return False
        
        # framing
        # error detection



        # Creates and starts receiver at a different thread
        # This is necessary for the asynchronous socket implementation
        receiver = rp.Receptor()
        receiver_trd = threading.Thread(target=receiver.start_server, daemon=True)
        receiver_trd.start()
        time.sleep(0.5)

        # if not tm.send_server()


if __name__ == "__main__":
    win = Programa()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()