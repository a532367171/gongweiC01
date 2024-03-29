﻿using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace gongweiC01ConsoleApp2
{
    class Program
    {
        static string dir, modelFile, labelsFile;

        static void Main(string[] args)
        {
                dir = "tmp";
                List<string> files = Directory.GetFiles("img").ToList();
                ModelFiles(dir);
                var graph = new TFGraph();
                // 从文件加载序列化的GraphDef
                var model = File.ReadAllBytes(modelFile);
                //导入GraphDef
                graph.Import(model, "");
                using (var session = new TFSession(graph))
                {
                    var labels = File.ReadAllLines(labelsFile);
                    Console.WriteLine("TensorFlow图像识别 LineZero");
                    foreach (var file in files)
                    {
                        // Run inference on the image files
                        // For multiple images, session.Run() can be called in a loop (and
                        // concurrently). Alternatively, images can be batched since the model
                        // accepts batches of image data as input.
                        var tensor = CreateTensorFromImageFile(file);

                        var runner = session.GetRunner();
                        runner.AddInput(graph["input"][0], tensor).Fetch(graph["output"][0]);
                        var output = runner.Run();
                        // output[0].Value() is a vector containing probabilities of
                        // labels for each image in the "batch". The batch size was 1.
                        // Find the most probably label index.

                        var result = output[0];
                        var rshape = result.Shape;
                        if (result.NumDims != 2 || rshape[0] != 1)
                        {
                            var shape = "";
                            foreach (var d in rshape)
                            {
                                shape += $"{d} ";
                            }
                            shape = shape.Trim();
                            Console.WriteLine($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
                            Environment.Exit(1);
                        }

                        // You can get the data in two ways, as a multi-dimensional array, or arrays of arrays, 
                        // code can be nicer to read with one or the other, pick it based on how you want to process
                        // it
                        bool jagged = true;

                        var bestIdx = 0;
                        float p = 0, best = 0;

                        if (jagged)
                        {
                            var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                            for (int i = 0; i < probabilities.Length; i++)
                            {
                                if (probabilities[i] > best)
                                {
                                    bestIdx = i;
                                    best = probabilities[i];
                                }
                            }

                        }
                        else
                        {
                            var val = (float[,])result.GetValue(jagged: false);

                            // Result is [1,N], flatten array
                            for (int i = 0; i < val.GetLength(1); i++)
                            {
                                if (val[0, i] > best)
                                {
                                    bestIdx = i;
                                    best = val[0, i];
                                }
                            }
                        }

                        Console.WriteLine($"{Path.GetFileName(file)} 最佳匹配: [{bestIdx}] {best * 100.0}% 标识为：{labels[bestIdx]}");
                    }
                }
                Console.ReadKey();
            }

            // Convert the image in filename to a Tensor suitable as input to the Inception model.
            static TFTensor CreateTensorFromImageFile(string file)
            {
                var contents = File.ReadAllBytes(file);

                // DecodeJpeg uses a scalar String-valued tensor as input.
                var tensor = TFTensor.CreateString(contents);

                TFGraph graph;
                TFOutput input, output;

                // Construct a graph to normalize the image
                ConstructGraphToNormalizeImage(out graph, out input, out output);

                // Execute that graph to normalize this one image
                using (var session = new TFSession(graph))
                {
                    var normalized = session.Run(
                             inputs: new[] { input },
                             inputValues: new[] { tensor },
                             outputs: new[] { output });

                    return normalized[0];
                }
            }

            // The inception model takes as input the image described by a Tensor in a very
            // specific normalized format (a particular image size, shape of the input tensor,
            // normalized pixel values etc.).
            //
            // This function constructs a graph of TensorFlow operations which takes as
            // input a JPEG-encoded string and returns a tensor suitable as input to the
            // inception model.
            static void ConstructGraphToNormalizeImage(out TFGraph graph, out TFOutput input, out TFOutput output)
            {
                // Some constants specific to the pre-trained model at:
                // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
                //
                // - The model was trained after with images scaled to 224x224 pixels.
                // - The colors, represented as R, G, B in 1-byte each were converted to
                //   float using (value - Mean)/Scale.

                const int W = 224;
                const int H = 224;
                const float Mean = 117;
                const float Scale = 1;

                graph = new TFGraph();
                input = graph.Placeholder(TFDataType.String);

                output = graph.Div(
                    x: graph.Sub(
                        x: graph.ResizeBilinear(
                            images: graph.ExpandDims(
                                input: graph.Cast(
                                    graph.DecodeJpeg(contents: input, channels: 3), DstT: TFDataType.Float),
                                dim: graph.Const(0, "make_batch")),
                            size: graph.Const(new int[] { W, H }, "size")),
                        y: graph.Const(Mean, "mean")),
                    y: graph.Const(Scale, "scale"));
            }

            /// <summary>
            /// 下载初始Graph和标签
            /// </summary>
            /// <param name="dir"></param>
            static void ModelFiles(string dir)
            {
                string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";

                modelFile = Path.Combine(dir, "tensorflow_inception_graph.pb");
                labelsFile = Path.Combine(dir, "imagenet_comp_graph_label_strings.txt");
                var zipfile = Path.Combine(dir, "inception5h.zip");

                if (File.Exists(modelFile) && File.Exists(labelsFile))
                    return;

                Directory.CreateDirectory(dir);
                var wc = new WebClient();
                wc.DownloadFile(url, zipfile);
                ZipFile.ExtractToDirectory(zipfile, dir);
                File.Delete(zipfile);
            }
    }
}
