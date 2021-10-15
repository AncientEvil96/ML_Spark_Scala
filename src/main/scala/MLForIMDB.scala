import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.ArrayBuffer

object MLForIMDB {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Test DataFrame").getOrCreate()

    val training = spark.createDataFrame(Seq(
      ("a b c d e spark", 1.0),
      ("b d", 0.0),
      ("spark f g h", 1.0),
      ("hadoop mapreduce", 0.0)
    )).toDF("text", "label")



    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(remover.getOutputCol())
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    var components = new ArrayBuffer[PipelineStage]
    components += tokenizer
    components += remover
    components += hashingTF
    components += lr

    val pipeline = new Pipeline().setStages(components.toArray);

  }

//  package com.tencent.angel.spark.automl.feature.preprocess
//
//  import org.apache.spark.ml.PipelineStage
//  import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer}
//  import org.apache.spark.sql.DataFrame
//
//  import scala.collection.mutable.ArrayBuffer
//
//  object Components {
//
//    def sample(data: DataFrame,
//               fraction: Double): DataFrame = {
//      data.sample(false, fraction)
//    }
//
//    def addSampler(components: ArrayBuffer[PipelineStage],
//                   inputCol: String,
//                   fraction: Double): Unit = {
//      val sampler = new Sampler(fraction)
//        .setInputCol("features")
//      components += sampler
//    }
//
//    def addTokenizer(components: ArrayBuffer[PipelineStage],
//                     inputCol: String,
//                     outputCol: String): Unit = {
//      val tokenizer = new Tokenizer()
//        .setInputCol(inputCol)
//        .setOutputCol(outputCol)
//      components += tokenizer
//    }
//
//    def addStopWordsRemover(components: ArrayBuffer[PipelineStage],
//                            inputCol: String,
//                            outputCol: String): Unit = {
//      val remover = new StopWordsRemover()
//        .setInputCol(inputCol)
//        .setOutputCol(outputCol)
//      components += remover
//    }
//
//  }



  //    val training = spark.createDataFrame(Seq(
  //    (0L, "a b c d e spark", 1.0),
  //    (1L, "b d", 0.0),
  //    (2L, "spark f g h", 1.0),
  //    (3L, "hadoop mapreduce", 0.0)
  //  )).toDF("id", "text", "label")
  //
  //  val tokenizer = new Tokenizer()
  //    .setInputCol("text")
  //    .setOutputCol("words")
  //  val hashingTF = new HashingTF()
  //    .setNumFeatures(1000)
  //    .setInputCol(tokenizer.getOutputCol())
  //    .setOutputCol("features")
  //  val stopwordsremover = new StopWordsRemover()
  //    .setInputCol("raw")
  //    .setOutputCol("filtered")
  //  val lr = new LogisticRegression()
  //    .setMaxIter(10)
  //    .setRegParam(0.001)
  //  val pipeline = new Pipeline()
  //    .setStages(new PipelineStage[] {
  //      tokenizer
  //      , hashingTF
  //      , lr
  //    });
  //
  //  val model = pipeline.fit(training);
  //
  //  val test = spark.createDataFrame(Seq(
  //    (4L, "spark i j k"),
  //    (5L, "l m n"),
  //    (6L, "spark hadoop spark"),
  //    (7L, "apache hadoop")
  //  )).toDF("id", "text")
  //
  //  model.transform(test)
  //    .select("id", "text", "probability", "prediction")
  //    .collect()
  //    .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
  //      println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  //    }
}