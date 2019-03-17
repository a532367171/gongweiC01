using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;

namespace gongweiC01ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var session = new TFSession())
            {
                var graph = session.Graph;
                Console.WriteLine(TFCore.Version);
                var a = graph.Const(2);
                var b = graph.Const(3);
                Console.WriteLine("a=2 b=3");

                // 两常量加
                var addingResults = session.GetRunner().Run(graph.Add(a, b));
                var addingResultValue = addingResults.GetValue();
                Console.WriteLine("a+b={0}", addingResultValue);

                // 两常量乘
                var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
                var multiplyResultValue = multiplyResults.GetValue();
                Console.WriteLine("a*b={0}", multiplyResultValue);
                var tft = new TFTensor(Encoding.UTF8.GetBytes($"Hello TensorFlow Version {TFCore.Version}! LineZero"));
                var hello = graph.Const(tft);
                var helloResults = session.GetRunner().Run(hello);
                Console.WriteLine(Encoding.UTF8.GetString((byte[])helloResults.GetValue()));
            }
            Console.ReadKey();
        }
    }
}
