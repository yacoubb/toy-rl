///<reference path="./node_modules/@types/p5/global.d.ts" />

let brushSize = 100;

setup = () => {
	createCanvas(512, 512);
	pixelDensity(1);
	background(0);
};

function draw() {
	if (mouseIsPressed) {
		noStroke();
		fill(255, 10);
		circle(mouseX, mouseY, brushSize);
	}
}

function mouseReleased() {}

windowResized = () => {
	// resizeCanvas(windowWidth, windowHeight);
};

function keyPressed(key) {
	console.log(key.code);
	if (key.code == 'KeyB') {
		filter(BLUR, 2);
	}
	if (key.code === 'Space') {
		// convert brightness to hsl values
		loadPixels();
		colorMode('hsb');
		console.log(pixels.length / 4);
		for (let x = 0; x < windowWidth; x++) {
			for (let y = 0; y < windowHeight; y++) {
				let val = pixels[(x + y * windowWidth) * 4];
				stroke(255 - val, 255, 255);
				point(x, y);
			}
		}
	}
	console.log('done');
}
