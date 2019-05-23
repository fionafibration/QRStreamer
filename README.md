# QRStreamer

Companion program to [TXQRAndroid](https://github.com/ThePlasmaRailgun/TXQR-Android)

Streams files to a set of QR codes and a GIF of those QR codes, to be decoded by the Android app. 

### Installation
Run `pip install -U qrstreamer` to install the command-line tools. Make sure your pip scripts directory is in the operating system PATH

### Usage

Run as `qrstreamer --help` to see usage information

### Credits

Uses a slightly modified port of [anrosent's LT coding library](https://github.com/anrosent/LT-Code) to perform LT coding. Inspired by (but incompatible with) [divan's TXQR for iPhones](https://github.com/divan/txqr)