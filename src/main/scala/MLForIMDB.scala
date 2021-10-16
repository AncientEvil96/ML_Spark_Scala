import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer


object MLForIMDB {
  def main(args: Array[String]): Unit = {
    val tr_input = "Train.csv"
    val ts_input = "Valid.csv"
    //    val tr_input = args(0)
    //    val ts_input = args(1)
    //    val output = args(2)

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("Test DataFrame")
      .getOrCreate()

    //    val training = spark.createDataFrame(Seq(
    //      ("a b c d e spark", 1.0),
    //      ("b d", 0.0),
    //      ("spark f g h", 1.0),
    //      ("hadoop mapreduce", 0.0)
    //    )).toDF("text", "label")

    val training = spark.read
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .csv(tr_input)
      .withColumn("label", col("label").cast(DataTypes.FloatType))

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val components = new ArrayBuffer[PipelineStage]
    components += tokenizer
    components += remover
    components += hashingTF
    components += lr

    val pipeline = new Pipeline().setStages(components.toArray)

    val model = pipeline.fit(training)

    //    var test = spark.createDataFrame(Seq(
    //      ("spark i j k",0),
    //      ("l m n",0),
    //      ("spark hadoop spark",0),
    //      ("apache hadoop",0)
    //    )).toDF("text","label")
    //
    //    test = test.select("text")

    val test = spark.read
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .csv(ts_input)
      .withColumn("label", col("label").cast(DataTypes.FloatType))

    //    model.transform(test)
    //      .select("id", "text", "probability", "prediction").limit(10)
    //      .collect()
    //      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    //        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
    //      }

//    model.transform(test)
//      .select("text", "probability", "prediction")
//      .limit(10)
//      .show()
    //      .collect()
    //      .foreach { case Row(text: String, prob: Vector, prediction: Double) =>
    //          println(s"($text) --> prob=$prob, prediction=$prediction")
    //      }
    //          .repartition(1) //сделаем 1 партицию вместо 200
    //          .write.mode("overwrite")
    //          .option("sep", ";") // поменяли раздилитель
    //          .csv(output)
    //                  .collect()
    //                  .foreach { case Row(text: String, prob: Vector, prediction: Double) =>
    //                    println(s"($text) --> prob=$prob, prediction=$prediction")
    //                  }

//    val accuracy = new MulticlassMetrics(predictions.zip(labels)).accuracy
    spark.stop()
  }
}

//docker cp target/scala-2.11/scala_spark_ml_2.11-0.1.jar gbhdp:/home/hduser/
//
//spark-submit --class MLForIMDB --master yarn --deploy-mode cluster scala_spark_ml_2.11-0.1.jar /user/hduser/imdb/Train.csv /user/hduser/imdb/Test.csv
//
//hdfs dfs -ls /user/hduser/ppkm-df-out
//hdfs dfs -cat /user/hduser/ppkm-df-out/*
//hdfs dfs -rm -r -skipTrash ppkm-df-out

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