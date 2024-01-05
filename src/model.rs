use std::fs;
use std::fs::{File, read};
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use actix_web::{delete, HttpResponse, web};
use serde::{Deserialize, Serialize};
use crate::{DATA_PATH, INTERPRET, MODEL_PATH, RESULT_PATH};

#[derive(Serialize, Deserialize)]
pub struct ModelRequest {
    train: String,
    test: String,
    valid: String,
    output_name: String,
    structure: String,
    seq_len: String,
    label_len: String,
    pred_len: String,
    e_layers: String,
    d_layers: String,
    factor: String,
    enc_in: String,
    dec_in: String,
    c_out: String,
    itr: String,
}

pub async fn handle_model_train(req: web::Json<ModelRequest>)
-> actix_web::Result<HttpResponse> {
    let mut command = Command::new(INTERPRET)
        .args(&[
            "./lib/libs/run.py",
            "--task_name",
            "long_term_forecast",
            "--is_training",
            "1",
            "--root_path",
            "./lib/upload_data",
            "--data_train_path",
            &format!("{}", req.train),
            "--data_test_path",
            &format!("{}", req.test),
            "--data_vali_path",
            &format!("{}", req.valid),
            "--model_id",
            &req.output_name,
            "--model",
            &req.structure,
            "--target",
            "press",
            "--features",
            "MS",
            "--seq_len",
            &req.seq_len.to_string(),
            "--label_len",
            &req.label_len.to_string(),
            "--pred_len",
            &req.pred_len.to_string(),
            "--e_layers",
            &req.e_layers.to_string(),
            "--d_layers",
            &req.d_layers.to_string(),
            "--factor",
            &req.factor,
            "--enc_in",
            &req.enc_in,
            "--dec_in",
            &req.dec_in,
            "--c_out",
            &req.c_out,
            "--des",
            "Exp",
            "--itr",
            &req.itr,
        ]).stdout(Stdio::piped())
        .spawn()?;

        println!("{:#?}", command);

    let stdout = command.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        println!("{}", line?);
    }

    let status = command.wait()?;

    println!("return");

    Ok(HttpResponse::Ok().body(""))
}

fn all_models() -> Vec<String>  {
    let entries = fs::read_dir(MODEL_PATH).unwrap();

    entries
        .map(|e| e
            .unwrap()
            .file_name()
            .to_str()
            .unwrap()
            .to_string())
        .collect::<Vec<_>>()
}

pub async fn handle_get_models() -> actix_web::Result<HttpResponse> {
    let names = all_models()
        .into_iter()
        .map(|e| e.split("_").map(|e| e.to_string())
            .collect::<Vec<_>>())
        .collect::<Vec<_>>();

    #[derive(Serialize)]
    struct Response {
        name: String,
        task_name: String,
        structure: String,
        data_source: String,
    }

    Ok(
        HttpResponse::Ok()
            .body(serde_json::to_string(
                &Response {
                    name: format!("{:?}", names
                        .iter()
                        .map(|e| e[3].clone())
                        .collect::<Vec<_>>()),
                    task_name: format!("{:?}", names
                        .iter()
                        .map(|_e| "longTermForecast".to_string())
                        .collect::<Vec<_>>()),
                    structure: format!("{:?}", names
                        .iter()
                        .map(|e| e[4].clone())
                        .collect::<Vec<_>>()),
                    data_source: format!("{:?}", names
                        .iter()
                        .map(|e| e[5].clone())
                        .collect::<Vec<_>>()),
                }
            )?)
    )
}

#[derive(Serialize, Deserialize)]
pub struct TestModelReq {
    name: String,
}

pub async fn handle_test_model(info: web::Json<TestModelReq>) -> actix_web::Result<HttpResponse> {
    let names = all_models();

    let dir_name = names.iter()
        .find(|e| e.find(&info.name).is_some()).unwrap();

    let title = format!("{} {} steps", dir_name.split("_").collect::<Vec<_>>()[3].clone(),
        &dir_name.split("_").collect::<Vec<_>>()[9].clone()[2..]);

    let command = Command::new(INTERPRET)
        .args(&["./lib/libs/calmse.py", "--directory", &RESULT_PATH.to_string(),
        "--a", dir_name, "--title", &title, "--test_csv", "./lib/upload_data/test.csv"])
        .output().unwrap();

    println!("{:#?}", command);

    let mut file = File::open("./tmp.png")?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let base64_string = base64::encode(&buffer);

    Ok(
        HttpResponse::Ok()
            .body(serde_json::to_string(&base64_string)?)
    )
}