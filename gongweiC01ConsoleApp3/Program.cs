using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace gongweiC01ConsoleApp3
{
    class Program
    {
        static string dir, modelFile, labelsFile;

        static void Main(string[] args)
        {
            var capture = new VideoCapture("rtmp://rtmp.open.ys7.com/openlive/61e96da9f12a4d058f4737d02c42998d");
            modelFile = "logs_2\\pb\\frozen_model.pb";
            //dir = "tmp";
            //List<string> files = Directory.GetFiles("img").ToList();
            //ModelFiles(dir);
            var graph = new TFGraph();
            // 从文件加载序列化的GraphDef
            var model = File.ReadAllBytes(modelFile);
            //导入GraphDef
            graph.Import(model, "");
            using (var windowSrc = new Window("src"))
            using (var frame = new Mat())
            using (var image缩小 = new Mat())
            using (var session = new TFSession(graph))
            {
                string file = "1.jpg";
                //var labels = File.ReadAllLines(labelsFile);
                Console.WriteLine("TensorFlow图像识别 LineZero");

                //var frame = new Mat();
                //var inrange = new Mat();
                //var fg = new Mat();

                while (true)
                {
                    capture.Read(frame);
                    if (frame.Empty())
                        break;
                    Cv2.Resize(frame, image缩小, new Size(280, 280), 0, 0, InterpolationFlags.Linear);//缩小28*28

                    Cv2.ImWrite(file, image缩小);

                    var tensor = CreateTensorFromImageFile(file);



                    // Run inference on the image files
                    // For multiple images, session.Run() can be called in a loop (and
                    // concurrently). Alternatively, images can be batched since the model
                    // accepts batches of image data as input.

                    var runner = session.GetRunner();
                    runner.AddInput(graph["x_input"][0], tensor).Fetch(graph["softmax_linear/softmax_linear"][0]);
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
                        double[] d = floatTodouble(probabilities);
                        double[] retResult = Softmax(d);

                        for (int i = 0; i < retResult.Length; i++)
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

                    //Console.WriteLine($"{Path.GetFileName(file)} 最佳匹配: [{bestIdx}] {best * 100.0}% 标识为：{labels[bestIdx]}");
                    string 标识1="";
                    switch (bestIdx)
                    {
                        case 0:
                            标识1= "kong0";
                            break;
                        case 1:
                            标识1 = "yao1";
                            break;
                        case 2:
                            标识1 = "yao2";
                            break;
                        case 3:
                            标识1 = "yao3";
                            break;
                        case 4:
                            标识1 = "yao4";
                            break;
                        case 5:
                            标识1 = "xian1";
                            break;
                        case 6:
                            标识1 = "xian2";
                            break;
                        case 7:
                            标识1 = "have7";
                            break;
                    }
                    string 标识2 = "--: "+(best ).ToString()+"%";

                    Point textPos = new Point(1,100);

                    image缩小.PutText(标识1+ 标识2, textPos, HersheyFonts.HersheySimplex, 0.5, Scalar.White);


                    windowSrc.ShowImage(image缩小);
                    Cv2.WaitKey(1000);
                }
            }
            Console.ReadKey();
        }

        private static double[] floatTodouble(float[] probabilities)
        {
            double[] DOU=new double[8];
            for (int i = 0; i < probabilities.Length; i++)
            {
                DOU[i] = (double)probabilities[i];
            }
            return DOU;
        }

        private static double[] Softmax(double[] probabilities)
        {
            double max = 0;
            double sum = 0;
            for (int i = 0; i <8; i++)
                if (max < probabilities[i])
                    max = probabilities[i];
            //#pragma omp parallel for  
            for (int i = 0; i < 8; i++)
            {
                probabilities[i] = Math.Exp(probabilities[i] - max);//防止数据溢出
                sum += probabilities[i];
            }
            //#pragma omp parallel for  
            for (int i = 0; i < 8; i++)
                probabilities[i] /= sum;

            return probabilities;
        }



        //private static TFTensor CreateTensorFromImageMat(Mat mat)
        //{

        //    TFGraph graph;
        //    TFOutput input, output;





        //}

        static TFTensor CreateTensorFromImageFile(string file)
        {
            var contents = File.ReadAllBytes(file);

            // DecodeJpeg uses a scalar String-valued tensor as input.
            var tensor = TFTensor.CreateString(contents);

            TFGraph graph;
            TFOutput input, output;

            // Construct a graph to normalize the image 归一化
            ConstructGraphToNormalizeImage(out graph, out input, out output);

            // Execute that graph to normalize this one image 执行图规范化这个形象
            using (var session = new TFSession(graph))
            {
                var normalized = session.Run(
                         inputs: new[] { input },
                         inputValues: new[] { tensor },
                         outputs: new[] { output });

                return normalized[0];
            }
        }

        static void ConstructGraphToNormalizeImage(out TFGraph graph, out TFOutput input, out TFOutput output)
        {
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained after with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.

            const int W = 208;
            const int H = 208;
            const float Mean = 0;
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



    }
}
