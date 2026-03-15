<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class WelcomeController extends Controller
{
    /**
     * Menampilkan halaman homepage.
     */
    public function index()
    {
        return view('welcome');
    }

    /**
     * Menampilkan halaman about us.
     */
    public function about()
    {
        return view('about-us');
    }
    public function howItWorks()
{
    return view('how-it-works');
}
}