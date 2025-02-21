// offscreen-worker.js

self.addEventListener('message', async (e) => {
  // Očekujemo da e.data bude data URL slike, npr. "data:image/jpeg;base64,..."
  const imageDataUrl = e.data;
  try {
    // Pretvori data URL u Blob, zatim u ImageBitmap
    const response = await fetch(imageDataUrl);
    const blob = await response.blob();
    const imageBitmap = await createImageBitmap(blob);

    // Kreiraj OffscreenCanvas s dimenzijama slike
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d');

    // Iscrtaj ImageBitmap na canvas
    ctx.drawImage(imageBitmap, 0, 0);

    // Ovde možete dodati dodatnu obradu (npr. filtriranje, skaliranje, itd.)
    // ...

    // Prenesi ImageBitmap nazad na glavni thread (transfer objekat)
    const outputBitmap = canvas.transferToImageBitmap();
    self.postMessage({ imageBitmap: outputBitmap }, [outputBitmap]);
  } catch (err) {
    console.error('Worker error:', err);
    self.postMessage({ error: err.toString() });
  }
});
