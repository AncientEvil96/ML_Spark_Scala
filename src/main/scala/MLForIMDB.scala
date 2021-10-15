import org.apache.spark
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType

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

    steps: => Seq[PipelineStage]


    val pipeline = new Pipeline().setStages(steps.toArray);

  }
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