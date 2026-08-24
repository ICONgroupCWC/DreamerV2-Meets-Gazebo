scene.remove("tx_1"), scene.remove("tx_2"), scene.remove("tx_3"), scene.remove(
    "tx_4"
), scene.remove("tx_5")

scene.tx_array = PlanarArray(
    num_rows=2,
    num_cols=4,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="V",
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

tx1 = Transmitter(
    name="tx_1",
    #  position=[-56, 2, 22],
    position=[-56, 50, 22],
)

tx2 = Transmitter(
    name="tx_2",
    #  position=[44.5, -1, 30]
    position=[50, 10, 33],
)

tx3 = Transmitter(
    name="tx_3",
    #  position=[0, 63, 18]
    position=[25.8, 70, 22],
)

tx4 = Transmitter(
    name="tx_4",
    #  position=[-21, -42, 10.5]
    position=[-30, -52, 22],
)

tx5 = Transmitter(
    name="tx_5",
    #  position=[-21, -42, 10.5]
    position=[-18.2, 11.5, 30.5],
)

scene.add(tx1)
scene.add(tx2)
scene.add(tx3)
scene.add(tx4)
scene.add(tx5)

scene.frequency = 2.14e9
scene.synthetic_array = True
subcarrier_spacing = 20e6 / 16  # 15e3
fft_size = 16
frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)

channels = []

for ep in tqdm(range(loc.shape[0])):

    for kk in range(loc[ep].shape[0]):
        rx = Receiver(
            name="rx_" + str(ep) + "_" + str(kk),
            position=[loc[ep, kk, 0] / 5 - 15, loc[ep, kk, 1] / 5 + 15, 2],
            orientation=[0, 0, 0],
        )

        scene.add(rx)

    tx1.look_at(rx), tx2.look_at(rx)

    paths = scene.compute_paths(max_depth=5, num_samples=1e6)

    a, tau = paths.cir()

    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)

    channels.append(h_freq.numpy().squeeze())

    for kk in range(loc[ep].shape[0]):
        scene.remove("rx_" + str(ep) + "_" + str(kk))
