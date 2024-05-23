module PE_array#(parameter num = 4)
(
	// interface to system
    input wire clk,
    input wire reset,
    input wire c_en,                              // compute enable
    input wire p_en,                              // preload enable
    // interface to PE row .....

    input wire signed[31:0]in_weight[num-1:0],             // wire from weight buffer direction

    input wire signed[31:0]in_input[num-1:0], 
                 // wire from input buffer direction
    output signed[31:0]result[num-1:0],

    output compute_finished

	);
    // some hint and suggestions but not compulsory:
    //1. you can use a set of shift resigters for data alignment before computing.
    //2. you can use  combinational circuits as the transpose unit after you
    // get the complete matrix, due to its simplicity.Maybe extra storage is needed.
    //3. you can use shift logic to offload the result from the PE_array.
    //4. only if the PE_array's state is computing ,the data should be aligned in parallelogram,
    //otherwise, it should be aligned in rectangle.
    //5. you need to calculte the compute_finished signal taking the transpose and computing etc into consideration.

    // Spec
    // input: input data belonging to the same row/column of an input matrix from weight/input buffer should arrive at PE array at the same time
    // output: input data belonging to the same row/column of an output matrix should leave PE array at the same time


    // to do: insert buffers in front of different rows or columns of PE array to ensure logic correction
    reg [num+2:0] c_cnt,p_cnt,o_cnt;
    reg o_en;
    reg com_en;
    reg signed[num-1:0] pre_en;

    reg signed[31:0] input_buffer[num-1:0][num-1:0];
    reg signed[31:0] weight_buffer[num-1:0][num-1:0];
    reg signed[31:0] result_buffer[num-1:0][num-1:0];

    reg signed[31:0] mid_input_buffer[num-1:0][num-1:0];
    reg signed[31:0] mid_weight_buffer[num-1:0][num-1:0];

    assign 
    assign flow_en=com_en&!finished;
    assign result=(o_cnt-1<num)?result_buffer[o_cnt-1]:result_buffer[0];
    assign 

    always@(posedge clk)begin
        com_en<=c_en;
        if(!reset)begin
            finished<=0;
        end
    end

    always@(posedge clk)begin
        if(p_en&reset)begin
            p_cnt<=p_cnt+1;
        end
        else begin
            p_cnt<=0;
        end
    end

    always@(posedge clk)begin
        if(c_en&reset)begin
            if(c_cnt==3*num-2)begin
                finished<=1;
            end
            c_cnt<=c_cnt+1;
        end
        else begin
            c_cnt<=0;
        end
    end

    always@(posedge clk)begin
        if(finished&reset)begin
            if(f_cnt==num-1)begin
                finished<=0;
            end
            f_cnt<=f_cnt+1;
        end
        else begin
            f_cnt<=0;
        end
    end

    integer a,b;
    always@(posedge clk)begin
        for(a=0;a<num;a=a+1)begin:intobuffer
            input_buffer[c_cnt][a]<=in_input[a];
            weight_buffer[a][c_cnt]<=in_weight[a];
        end
    end

    // to do: some glue jobs


    wire signed[31:0] pe2mid_input[num-1:0][num-1:0];
    wire signed[31:0] mid2pe_input[num-1:0][num-1:0];
    wire signed[31:0] pe2mid_weight[num-1:0][num-1:0];
    wire signed[31:0] mid2pe_weight[num-1:0][num-1:0];

    
    assign mid_input_buffer=pe2mid_input;
    assign mid_weight_buffer=pe2mid_weight;

    always@(posedge clk)begin
        mid2pe_input<=mid_input_buffer;
        mid2pe_weight<=mid_weight_buffer;
    end

    wire signed[31:0] first_input[num-1:0];
    wire signed[31:0] first_weight[num-1:0];

    assign first_input=input_buffer[0];
    assign first_weight=weight_buffer[0];
    

    genvar i,j;
    generate
        for(i=0;i<num;i=i+1)
        begin:genrow
            for(j=0;j<num;j=j+1)
            begin:genpe
                if(i>0&&j>0)begin
                    PE pe(
                        .clk(clk),
                        .reset(reset),
                        .c_en(com_en),
                        .p_en(pre_en[i]),
                        .in_weight(mid2pe_weight[i][j-1]),
                        .in_input(mid2pe_input[i-1][j]),
                        .in_preload(in_input[j]),
                        .flow_output(pe2mid_input[i][j]),
                        .flow_weight(pe2mid_weight[i][j]),
                        .result(result_buffer[i][j])
                    );
                end
                else if(i==0&&j==0)begin
                    PE pe(
                        .clk(clk),
                        .reset(reset),
                        .c_en(com_en),
                        .p_en(pre_en[i]),
                        .in_weight(first_weight[i][j]),
                        .in_input(first_input[i][j]),
                        .in_preload(in_input[j]),
                        .flow_output(pe2mid_input[i][j]),
                        .flow_weight(pe2mid_weight[i][j]),
                        .result(result_buffer[i][j])
                    );
                end
                else if(i==0)begin
                    PE pe(
                        .clk(clk),
                        .reset(reset),
                        .c_en(com_en),
                        .p_en(pre_en[i]),
                        .in_weight(mid2pe_weight[i][j-1]),
                        .in_input(first_input[i][j]),
                        .in_preload(in_input[j]),
                        .flow_output(pe2mid_input[i][j]),
                        .flow_weight(pe2mid_weight[i][j]),
                        .result(result_buffer[i][j])
                    );
                end
                else if(j==0)begin
                    PE pe(
                        .clk(clk),
                        .reset(reset),
                        .c_en(com_en),
                        .p_en(pre_en[i]),
                        .in_weight(first_weight[i][j]),
                        .in_input(mid2pe_input[i-1][j]),
                        .in_preload(in_input[j]),
                        .flow_output(pe2mid_input[i][j]),
                        .flow_weight(pe2mid_weight[i][j]),
                        .result(result_buffer[i][j]) 
                    );
                end 
            end
        end
    endgenerate

endmodule
