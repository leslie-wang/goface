package main

import (
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"os"
	"strconv"

	"github.com/jdeng/goface"
	"github.com/oliamb/cutter"
)

func main() {
	imgFile := flag.String("input", "1.jpg", "input jpeg file")
	flag.Parse()

	bs, err := ioutil.ReadFile(*imgFile)
	if err != nil {
		log.Fatal(err)
	}

	img, err := goface.TensorFromJpeg(bs)
	if err != nil {
		log.Fatal(err)
	}

	det, err := goface.NewMtcnnDetector("../cmd/mtcnn.pb")
	if err != nil {
		log.Fatal(err)
	}
	defer det.Close()

	// 0 for default
	det.Config(0, 0, []float32{0.7, 0.7, 0.95})

	bbox, err := det.DetectFaces(img)
	if err != nil {
		log.Fatal(err)
	}

	if len(bbox) == 0 {
		log.Println("No face found")
		return
	}

	log.Printf("%d faces found in %s: %v\n", len(bbox), *imgFile, bbox)

	f, err := os.Open(*imgFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	imgSrc, err := jpeg.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	var margin float32 = 16.0
	for i, box := range bbox {
		box[0] -= margin
		box[1] -= margin
		box[2] += margin
		box[3] += margin

		fo, err := os.Create(strconv.Itoa(i) + ".png")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("--- %d crop\n", i)
		cImg, err := cutter.Crop(imgSrc, cutter.Config{
			Height:  int(box[3] - box[1] + 10),                       // height in pixel or Y ratio(see Ratio Option below)
			Width:   int(box[2] - box[0] + 10),                       // width in pixel or X ratio
			Mode:    cutter.TopLeft,                                  // Accepted Mode: TopLeft, Centered
			Anchor:  image.Point{int(box[0] - 10), int(box[1] - 10)}, // Position of the top left point
			Options: 0,                                               // Accepted Option: Ratio
		})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("--- %d encode\n", i)
		if err := png.Encode(fo, cImg); err != nil {
			log.Fatal(err)
		}
		if err := fo.Close(); err != nil {
			log.Fatal(err)
		}
	}

}
