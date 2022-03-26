let phrase=document.querySelector(".input-inputText");
let GPT2=document.querySelector(".GPT2-model");
let T5=document.querySelector(".T5-model");
let rephrase=document.querySelector(".btn-rephrase");
let textPraphrasing=document.querySelector(".div-textPraphrasing");

let mode ="GPT2"


function chnageModeToT5(){
    mode="T5";
    T5.style.color="brown";
    GPT2.style.color="black";
    GPT2.style.borderBottom="none";
    T5.style.borderBottom="solid brown 2px";
}
function chnageModeToGPT2(){
    mode="GPT2";
    GPT2.style.color="brown";
    T5.style.color="black";
    T5.style.borderBottom="none";
    GPT2.style.borderBottom="solid brown 2px";
}

T5.addEventListener("click",chnageModeToT5);
GPT2.addEventListener("click",chnageModeToGPT2);


rephrase.onclick=function(){
    var xhr=new XMLHttpRequest();

    if (phrase.value.length<1){
        alert("write samething before click in btn");
    }else{
        var data=JSON.stringify(data);
        xhr.open("GET","/paraphrase?model="+mode+"&text="+phrase.value);
        xhr.send();

        xhr.onreadystatechange=function(){
            if(this.readyState==4 && this.status==200){  
                let res=JSON.parse(this.responseText)
                console.log(res["text"])
                textPraphrasing.innerHTML=res["text"]
            }
        }
        

    }
}